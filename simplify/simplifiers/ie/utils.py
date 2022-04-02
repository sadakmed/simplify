import itertools
import logging
import os
import warnings
from operator import itemgetter

import nltk
import numpy as np
from transformers.file_utils import cached_path, hf_bucket_url

SEP = ";;;"
QUESTION_TRG_INDEX = 3  # index of the predicate within the question
QUESTION_PP_INDEX = 5
QUESTION_OBJ2_INDEX = 6
DATA_CONFIG_NAME = "data-config.json"
WEIGHT_NAME = "model.pt"


def read_config_file(file_path):
    with open(file_path) as f:
        return json.loads(f.read())

def get_config(path):
    return _get_file(path, DATA_CONFIG_NAME)


def get_weight(path):
    return _get_file(path, WEIGHT_NAME)


def _get_file(path, file_name):
    possible_config_file = os.path.join(path, file_name)
    if os.path.isdir(path) and on.path.isfile(possible_config_file):
        config_file = possible_config_file
    elif os.path.isfile(path):
        config_file = path
    else:
        config_file = hf_bucket_url(path, filename=file_name)
    try:
        return cached_path(config_file)
    except Exception as e:
        raise f"can't load {path}, due to {str(e)}"




def process_extraction(extraction, sentence, score):
    # rel, arg1, arg2, loc, time = [], [], [], [], []
    rel, arg1, arg2, loc_time, args = [], [], [], [], []
    tag_mode = "none"
    rel_case = 0
    for i, token in enumerate(sentence):
        if "[unused" in token:
            if extraction[i].item() == 2:
                rel_case = int(re.search("\[unused(.*)\]", token).group(1))
            continue
        if extraction[i] == 1:
            arg1.append(token)
        if extraction[i] == 2:
            rel.append(token)
        if extraction[i] == 3:
            arg2.append(token)
        if extraction[i] == 4:
            loc_time.append(token)
    rel = " ".join(rel).strip()
    if rel_case == 1:
        rel = "is " + rel
    elif rel_case == 2:
        rel = "is " + rel + " of"
    elif rel_case == 3:
        rel = "is " + rel + " from"
    arg1 = " ".join(arg1).strip()
    arg2 = " ".join(arg2).strip()
    args = " ".join(args).strip()
    loc_time = " ".join(loc_time).strip()
    arg2 = (arg2 + " " + loc_time + " " + args).strip()
    sentence_str = " ".join(sentence).strip()
    extraction = Extraction(
        pred=rel, head_pred_index=None, sent=sentence_str, confidence=score, index=0
    )
    extraction.addArg(arg1)
    extraction.addArg(arg2)
    return extraction


def coords_to_sentences(conj_coords, words):

    for k in list(conj_coords):
        if conj_coords[k] is None:
            conj_coords.pop(k)

    for k in list(conj_coords):
        if words[conj_coords[k].cc] in ["nor", "&"]:  # , 'or']:
            conj_coords.pop(k)

    remove_unbreakable_conjuncts(conj_coords, words)

    conj_words = []
    for k in list(conj_coords):
        for conjunct in conj_coords[k].conjuncts:
            conj_words.append(" ".join(words[conjunct[0] : conjunct[1] + 1]))

    sentence_indices = []
    for i in range(0, len(words)):
        sentence_indices.append(i)

    roots, parent_mapping, child_mapping = get_tree(conj_coords)

    q = list(roots)

    sentences = []
    count = len(q)
    new_count = 0

    conj_same_level = []

    while len(q) > 0:

        conj = q.pop(0)
        count -= 1
        conj_same_level.append(conj)

        for child in child_mapping[conj]:
            q.append(child)
            new_count += 1

        if count == 0:
            get_sentences(sentences, conj_same_level, conj_coords, sentence_indices)
            count = new_count
            new_count = 0
            conj_same_level = []

    word_sentences = [
        " ".join([words[i] for i in sorted(sentence)]) for sentence in sentences
    ]

    return word_sentences, conj_words, sentences


def get_tree(conj):
    parent_child_list = []

    child_mapping, parent_mapping = {}, {}

    for key in conj:
        assert conj[key].cc == key
        parent_child_list.append([])
        for k in conj:
            if conj[k] is not None:
                if is_parent(conj[key], conj[k]):
                    parent_child_list[-1].append(k)

        child_mapping[key] = parent_child_list[-1]

    parent_child_list.sort(key=list.__len__)

    for i in range(0, len(parent_child_list)):
        for child in parent_child_list[i]:
            for j in range(i + 1, len(parent_child_list)):
                if child in parent_child_list[j]:
                    parent_child_list[j].remove(child)

    for key in conj:
        for child in child_mapping[key]:
            parent_mapping[child] = key

    roots = []
    for key in conj:
        if key not in parent_mapping:
            roots.append(key)

    return roots, parent_mapping, child_mapping


def is_parent(parent, child):
    min = child.conjuncts[0][0]
    max = child.conjuncts[-1][-1]

    for conjunct in parent.conjuncts:
        if conjunct[0] <= min and conjunct[1] >= max:
            return True
    return False


def get_sentences(sentences, conj_same_level, conj_coords, sentence_indices):
    for conj in conj_same_level:

        if len(sentences) == 0:

            for conj_structure in conj_coords[conj].conjuncts:
                sentence = []
                for i in range(conj_structure[0], conj_structure[1] + 1):
                    sentence.append(i)
                sentences.append(sentence)

            min = conj_coords[conj].conjuncts[0][0]
            max = conj_coords[conj].conjuncts[-1][-1]

            for sentence in sentences:
                for i in sentence_indices:
                    if i < min or i > max:
                        sentence.append(i)

        else:
            to_add = []
            to_remove = []

            for sentence in sentences:
                if conj_coords[conj].conjuncts[0][0] in sentence:
                    sentence.sort()

                    min = conj_coords[conj].conjuncts[0][0]
                    max = conj_coords[conj].conjuncts[-1][-1]

                    for conj_structure in conj_coords[conj].conjuncts:
                        new_sentence = []
                        for i in sentence:
                            if (
                                i in range(conj_structure[0], conj_structure[1] + 1)
                                or i < min
                                or i > max
                            ):
                                new_sentence.append(i)

                        to_add.append(new_sentence)

                    to_remove.append(sentence)

            for sent in to_remove:
                sentences.remove(sent)
            sentences.extend(to_add)


def remove_unbreakable_conjuncts(conj, words):

    unbreakable_indices = []

    unbreakable_words = [
        "between",
        "among",
        "sum",
        "total",
        "addition",
        "amount",
        "value",
        "aggregate",
        "gross",
        "mean",
        "median",
        "average",
        "center",
        "equidistant",
        "middle",
    ]

    for i, word in enumerate(words):
        if word.lower() in unbreakable_words:
            unbreakable_indices.append(i)

    to_remove = []
    span_start = 0

    for key in conj:
        span_end = conj[key].conjuncts[0][0] - 1
        for i in unbreakable_indices:
            if span_start <= i <= span_end:
                to_remove.append(key)
        span_start = conj[key].conjuncts[-1][-1] + 1

    for k in set(to_remove):
        conj.pop(k)


class Coordination(object):
    __slots__ = ("cc", "conjuncts", "seps", "label")

    def __init__(self, cc, conjuncts, seps=None, label=None):
        assert isinstance(conjuncts, (list, tuple)) and len(conjuncts) >= 2
        assert all(isinstance(conj, tuple) for conj in conjuncts)
        conjuncts = sorted(conjuncts, key=lambda span: span[0])
        # NOTE(chantera): The form 'A and B, C' is considered to be coordination.  # NOQA
        # assert cc > conjuncts[-2][1] and cc < conjuncts[-1][0]
        assert cc > conjuncts[0][1] and cc < conjuncts[-1][0]
        if seps is not None:
            # if len(seps) == len(conjuncts) - 2:
            #     for i, sep in enumerate(seps):
            #         assert conjuncts[i][1] < sep and conjuncts[i + 1][0] > sep
            # else:
            if len(seps) != len(conjuncts) - 2:
                warnings.warn(
                    "Coordination does not contain enough separators. "
                    "It may be a wrong coordination: "
                    "cc={}, conjuncts={}, separators={}".format(cc, conjuncts, seps)
                )
        else:
            seps = []
        self.cc = cc
        self.conjuncts = tuple(conjuncts)
        self.seps = tuple(seps)
        self.label = label

    def get_pair(self, index, check=False):
        pair = None
        for i in range(1, len(self.conjuncts)):
            if self.conjuncts[i][0] > index:
                pair = (self.conjuncts[i - 1], self.conjuncts[i])
                assert pair[0][1] < index and pair[1][0] > index
                break
        if check and pair is None:
            raise LookupError("Could not find any pair for index={}".format(index))
        return pair

    def __repr__(self):
        return "Coordination(cc={}, conjuncts={}, seps={}, label={})".format(
            self.cc, self.conjuncts, self.seps, self.label
        )

    def __eq__(self, other):
        if not isinstance(other, Coordination):
            return False
        return (
            self.cc == other.cc
            and len(self.conjuncts) == len(other.conjuncts)
            and all(
                conj1 == conj2 for conj1, conj2 in zip(self.conjuncts, other.conjuncts)
            )
        )


class Extraction:
    """
    Stores sentence, single predicate and corresponding arguments.
    """

    def __init__(
        self, pred, head_pred_index, sent, confidence, question_dist="", index=-1
    ):
        self.pred = pred
        self.head_pred_index = head_pred_index
        self.sent = sent
        self.args = []
        self.confidence = confidence
        self.matched = []
        self.questions = {}
        # self.indsForQuestions = defaultdict(lambda: set())
        self.is_mwp = False
        self.question_dist = question_dist
        self.index = index

    def distArgFromPred(self, arg):
        assert len(self.pred) == 2
        dists = []
        for x in self.pred[1]:
            for y in arg.indices:
                dists.append(abs(x - y))

        return min(dists)

    def argsByDistFromPred(self, question):
        return sorted(
            self.questions[question], key=lambda arg: self.distArgFromPred(arg)
        )

    def addArg(self, arg, question=None):
        self.args.append(arg)
        if question:
            self.questions[question] = self.questions.get(question, []) + [
                Argument(arg)
            ]

    def noPronounArgs(self):
        """
        Returns True iff all of this extraction's arguments are not pronouns.
        """
        for (a, _) in self.args:
            tokenized_arg = nltk.word_tokenize(a)
            if len(tokenized_arg) == 1:
                _, pos_tag = nltk.pos_tag(tokenized_arg)[0]
                if "PRP" in pos_tag:
                    return False
        return True

    def isContiguous(self):
        return all([indices for (_, indices) in self.args])

    def toBinary(self):
        """Try to represent this extraction's arguments as binary
        If fails, this function will return an empty list."""

        ret = [self.elementToStr(self.pred)]

        if len(self.args) == 2:
            # we're in luck
            return ret + [self.elementToStr(arg) for arg in self.args]

        return []

        if not self.isContiguous():
            # give up on non contiguous arguments (as we need indexes)
            return []

        # otherwise, try to merge based on indices
        # TODO: you can explore other methods for doing this
        binarized = self.binarizeByIndex()

        if binarized:
            return ret + binarized

        return []

    def elementToStr(self, elem, print_indices=True):
        """formats an extraction element (pred or arg) as a raw string
        removes indices and trailing spaces"""
        if print_indices:
            return str(elem)
        if isinstance(elem, str):
            return elem
        if isinstance(elem, tuple):
            ret = elem[0].rstrip().lstrip()
        else:
            ret = " ".join(elem.words)
        assert ret, "empty element? {0}".format(elem)
        return ret

    def binarizeByIndex(self):
        extraction = [self.pred] + self.args
        markPred = [(w, ind, i == 0) for i, (w, ind) in enumerate(extraction)]
        sortedExtraction = sorted(markPred, key=lambda ws, indices, f: indices[0])
        s = " ".join(
            [
                "{1} {0} {1}".format(self.elementToStr(elem), SEP)
                if elem[2]
                else self.elementToStr(elem)
                for elem in sortedExtraction
            ]
        )
        binArgs = [a for a in s.split(SEP) if a.rstrip().lstrip()]

        if len(binArgs) == 2:
            return binArgs

        # failure
        return []

    def bow(self):
        return " ".join([self.elementToStr(elem) for elem in [self.pred] + self.args])

    def getSortedArgs(self):
        """
        Sort the list of arguments.
        If a question distribution is provided - use it,
        otherwise, default to the order of appearance in the sentence.
        """
        if self.question_dist:
            # There's a question distribtuion - use it
            return self.sort_args_by_distribution()
        ls = []
        for q, args in self.questions.iteritems():
            if len(args) != 1:
                logging.debug("Not one argument: {}".format(args))
                continue
            arg = args[0]
            indices = list(self.indsForQuestions[q].union(arg.indices))
            if not indices:
                logging.debug("Empty indexes for arg {} -- backing to zero".format(arg))
                indices = [0]
            ls.append(((arg, q), indices))
        return [a for a, _ in sorted(ls, key=lambda _, indices: min(indices))]

    def question_prob_for_loc(self, question, loc):
        """
        Returns the probability of the given question leading to argument
        appearing in the given location in the output slot.
        """
        gen_question = generalize_question(question)
        q_dist = self.question_dist[gen_question]
        logging.debug("distribution of {}: {}".format(gen_question, q_dist))

        return float(q_dist.get(loc, 0)) / sum(q_dist.values())

    def sort_args_by_distribution(self):
        """
        Use this instance's question distribution (this func assumes it exists)
        in determining the positioning of the arguments.
        Greedy algorithm:
        0. Decide on which argument will serve as the ``subject'' (first slot) of this extraction
        0.1 Based on the most probable one for this spot
        (special care is given to select the highly-influential subject position)
        1. For all other arguments, sort arguments by the prevalance of their questions
        2. For each argument:
        2.1 Assign to it the most probable slot still available
        2.2 If non such exist (fallback) - default to put it in the last location
        """
        INF_LOC = 100  # Used as an impractical last argument

        # Store arguments by slot
        ret = {INF_LOC: []}
        logging.debug("sorting: {}".format(self.questions))

        # Find the most suitable arguemnt for the subject location
        logging.debug(
            "probs for subject: {}".format(
                [
                    (q, self.question_prob_for_loc(q, 0))
                    for (q, _) in self.questions.iteritems()
                ]
            )
        )

        subj_question, subj_args = max(
            self.questions.iteritems(),
            key=lambda q, _: self.question_prob_for_loc(q, 0),
        )

        ret[0] = [(subj_args[0], subj_question)]

        # Find the rest
        for (question, args) in sorted(
            [
                (q, a)
                for (q, a) in self.questions.iteritems()
                if (q not in [subj_question])
            ],
            key=lambda q, _: sum(self.question_dist[generalize_question(q)].values()),
            reverse=True,
        ):
            gen_question = generalize_question(question)
            arg = args[0]
            assigned_flag = False
            for (loc, count) in sorted(
                self.question_dist[gen_question].iteritems(),
                key=lambda _, c: c,
                reverse=True,
            ):
                if loc not in ret:
                    # Found an empty slot for this item
                    # Place it there and break out
                    ret[loc] = [(arg, question)]
                    assigned_flag = True
                    break

            if not assigned_flag:
                # Add this argument to the non-assigned (hopefully doesn't happen much)
                logging.debug(
                    "Couldn't find an open assignment for {}".format(
                        (arg, gen_question)
                    )
                )
                ret[INF_LOC].append((arg, question))

        logging.debug("Linearizing arg list: {}".format(ret))

        # Finished iterating - consolidate and return a list of arguments
        return [
            arg
            for (_, arg_ls) in sorted(ret.iteritems(), key=lambda k, v: int(k))
            for arg in arg_ls
        ]

    def __str__(self):
        pred_str = self.elementToStr(self.pred)
        return "{}\t{}\t{}".format(
            self.get_base_verb(pred_str),
            self.compute_global_pred(pred_str, self.questions.keys()),
            "\t".join(
                [
                    escape_special_chars(
                        self.augment_arg_with_question(self.elementToStr(arg), question)
                    )
                    for arg, question in self.getSortedArgs()
                ]
            ),
        )

    def get_base_verb(self, surface_pred):
        """
        Given the surface pred, return the original annotated verb
        """
        # Assumes that at this point the verb is always the last word
        # in the surface predicate
        return surface_pred.split(" ")[-1]

    def compute_global_pred(self, surface_pred, questions):
        """
        Given the surface pred and all instansiations of questions,
        make global coherence decisions regarding the final form of the predicate
        This should hopefully take care of multi word predicates and correct inflections
        """
        from operator import itemgetter

        split_surface = surface_pred.split(" ")

        if len(split_surface) > 1:
            # This predicate has a modal preceding the base verb
            verb = split_surface[-1]
            ret = split_surface[:-1]  # get all of the elements in the modal
        else:
            verb = split_surface[0]
            ret = []

        split_questions = map(lambda question: question.split(" "), questions)

        preds = map(
            normalize_element, map(itemgetter(QUESTION_TRG_INDEX), split_questions)
        )
        if len(set(preds)) > 1:
            # This predicate is appears in multiple ways, let's stick to the base form
            ret.append(verb)

        if len(set(preds)) == 1:
            # Change the predciate to the inflected form
            # if there's exactly one way in which the predicate is conveyed
            ret.append(preds[0])

            pps = map(
                normalize_element, map(itemgetter(QUESTION_PP_INDEX), split_questions)
            )

            obj2s = map(
                normalize_element, map(itemgetter(QUESTION_OBJ2_INDEX), split_questions)
            )

            if len(set(pps)) == 1:
                # If all questions for the predicate include the same pp attachemnt -
                # assume it's a multiword predicate
                self.is_mwp = (
                    True  # Signal to arguments that they shouldn't take the preposition
                )
                ret.append(pps[0])

        # Concat all elements in the predicate and return
        return " ".join(ret).strip()

    def augment_arg_with_question(self, arg, question):
        """
        Decide what elements from the question to incorporate in the given
        corresponding argument
        """
        # Parse question
        wh, aux, sbj, trg, obj1, pp, obj2 = map(
            normalize_element, question.split(" ")[:-1]
        )  # Last split is the question mark

        # Place preposition in argument
        # This is safer when dealing with n-ary arguments, as it's directly attaches to the
        # appropriate argument
        if (not self.is_mwp) and pp and (not obj2):
            if not (arg.startswith("{} ".format(pp))):
                # Avoid repeating the preporition in cases where both question and answer contain it
                return " ".join([pp, arg])

        # Normal cases
        return arg

    def clusterScore(self, cluster):
        """
        Calculate cluster density score as the mean distance of the maximum distance of each slot.
        Lower score represents a denser cluster.
        """
        logging.debug("*-*-*- Cluster: {}".format(cluster))

        # Find global centroid
        arr = np.array([x for ls in cluster for x in ls])
        centroid = np.sum(arr) / arr.shape[0]
        logging.debug("Centroid: {}".format(centroid))

        # Calculate mean over all maxmimum points
        return np.average([max([abs(x - centroid) for x in ls]) for ls in cluster])

    def resolveAmbiguity(self):
        """
        Heursitic to map the elments (argument and predicates) of this extraction
        back to the indices of the sentence.
        """
        ## TODO: This removes arguments for which there was no consecutive span found
        ## Part of these are non-consecutive arguments,
        ## but other could be a bug in recognizing some punctuation marks

        elements = [self.pred] + [(s, indices) for (s, indices) in self.args if indices]
        logging.debug("Resolving ambiguity in: {}".format(elements))

        # Collect all possible combinations of arguments and predicate indices
        # (hopefully it's not too much)
        all_combinations = list(itertools.product(*map(itemgetter(1), elements)))
        logging.debug("Number of combinations: {}".format(len(all_combinations)))

        # Choose the ones with best clustering and unfold them
        resolved_elements = zip(
            map(itemgetter(0), elements),
            min(all_combinations, key=lambda cluster: self.clusterScore(cluster)),
        )
        logging.debug("Resolved elements = {}".format(resolved_elements))

        self.pred = resolved_elements[0]
        self.args = resolved_elements[1:]

    def conll(self, external_feats={}):
        """
        Return a CoNLL string representation of this extraction
        """
        return (
            "\n".join(
                [
                    "\t".join(
                        map(
                            str,
                            [i, w]
                            + list(self.pred)
                            + [self.head_pred_index]
                            + external_feats
                            + [self.get_label(i)],
                        )
                    )
                    for (i, w) in enumerate(self.sent.split(" "))
                ]
            )
            + "\n"
        )

    def get_label(self, index):
        """
        Given an index of a word in the sentence -- returns the appropriate BIO conll label
        Assumes that ambiguation was already resolved.
        """
        # Get the element(s) in which this index appears
        ent = [
            (elem_ind, elem)
            for (elem_ind, elem) in enumerate(
                map(itemgetter(1), [self.pred] + self.args)
            )
            if index in elem
        ]

        if not ent:
            # index doesnt appear in any element
            return "O"

        if len(ent) > 1:
            # The same word appears in two different answers
            # In this case we choose the first one as label
            logging.warn(
                "Index {} appears in one than more element: {}".format(
                    index, "\t".join(map(str, [ent, self.sent, self.pred, self.args]))
                )
            )

        ## Some indices appear in more than one argument (ones where the above message appears)
        ## From empricial observation, these seem to mostly consist of different levels of granularity:
        ##     what	had	_	been taken	_	_	_	?	loan commitments topping $ 3 billion
        ##     how much	had	_	been taken	_	_	_	?	topping $ 3 billion
        ## In these cases we heuristically choose the shorter answer span, hopefully creating minimal spans
        ## E.g., in this example two arguemnts are created: (loan commitments, topping $ 3 billion)

        elem_ind, elem = min(ent, key=lambda _, ls: len(ls))

        # Distinguish between predicate and arguments
        prefix = "P" if elem_ind == 0 else "A{}".format(elem_ind - 1)

        # Distinguish between Beginning and Inside labels
        suffix = "B" if index == elem[0] else "I"

        return "{}-{}".format(prefix, suffix)

    def __str__(self):
        return "{0}\t{1}".format(
            self.elementToStr(self.pred, print_indices=True),
            "\t".join([self.elementToStr(arg) for arg in self.args]),
        )


class Argument:
    def __init__(self, arg):
        self.words = [x for x in arg[0].strip().split(" ") if x]
        self.posTags = map(itemgetter(1), nltk.pos_tag(self.words))
        self.indices = arg[1]
        self.feats = {}

    def __str__(self):
        return "({})".format(
            "\t".join(
                map(
                    str,
                    [(" ".join(self.words)).replace("\t", "\\t"), str(self.indices)],
                )
            )
        )


def normalize_element(elem):
    """
    Return a surface form of the given question element.
    the output should be properly able to precede a predicate (or blank otherwise)
    """
    return elem.replace("_", " ") if (elem != "_") else ""


## Helper functions
def escape_special_chars(s):
    return s.replace("\t", "\\t")


def generalize_question(question):
    """
    Given a question in the context of the sentence and the predicate index within
    the question - return a generalized version which extracts only order-imposing features
    """
    import nltk  # Using nltk since couldn't get spaCy to agree on the tokenization

    wh, aux, sbj, trg, obj1, pp, obj2 = question.split(" ")[
        :-1
    ]  # Last split is the question mark
    return " ".join([wh, sbj, obj1])
