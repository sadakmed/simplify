import json
import os
from collections import Counter

import nltk

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import (Coordination, coords_to_sentences, ext_to_triplet,
                   get_config, process_extraction)


def collate_pad_data(max_depth):
    max_depth = max_depth

    def wrapper(data):
        nonlocal max_depth
        return pad_data(data, max_depth)

    return wrapper


def pad_data(data, max_depth):
    input_ids = [ex["input_ids"] for ex in data]
    word_begin_index = [d["word_begin_index"] for d in data]
    sentence_index = [d["sentence_index"] for d in data]
    labels = [d["labels"] for d in data]
    max_length = max((len(i) for i in input_ids))
    input_ids = torch.tensor([pad_list(i, max_length, 0) for i in input_ids])
    max_length = max((len(i) for i in word_begin_index))
    word_begin_index = torch.tensor(
        [pad_list(i, max_length, 0) for i in word_begin_index]
    )
    sentence_index = torch.tensor(sentence_index)
    for i, label in enumerate(labels):
        to_add = max_depth - len(label)
        if to_add > 0:
            labels[i] = labels[i] + [[0] * len(label[0])] * to_add
    labels = torch.tensor(
        [[pad_list(l, max_length, -100) for l in label] for label in labels]
    )
    # each input_ids has multiple labels(targets) following the depth, here we make sure that all
    # the input_idss have the number of lables
    # NOTE: labels need to be padded
    padded = {
        "input_ids": input_ids,
        "labels": labels,
        "word_begin_index": word_begin_index,
        "sentence_index": sentence_index,
    }
    return dot_dict(padded)


def pad_list(list_, size, padding_token=0):
    to_add = size - len(list_)
    if to_add > 0:
        return list_ + [padding_token] * to_add
    if to_add == 0:
        return list_
    return list_[:-(-to_add)]  # truncate


def truncate(input_ids, word_begin_index, max_length):
    max_len = max_length - 2
    input_ids = input_ids[:max_len]
    unsqueeze_wb = [[i] * j for i, j in enumerate(word_begin_index)]
    flatten_uwb = [i for l in unsqueeze_wb for i in l]
    flatten_uwb = flatten_uwb[:max_len]
    word_begin_index = list(Counter(flatten_uwb).values())
    return input_ids, word_begin_index


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


ADDITIONAL_SPECIAL_TOKENS = [f"[unused{i}]" for i in range(1, 4)]


class Data:
    def __init__(self, config, *args, **kwargs):
        self.config = dot_dict(config) if isinstance(config, dict) else config
        self.config.additional_special_tokens = (
            self.config.additional_special_tokens or ADDITIONAL_SPECIAL_TOKENS
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            additional_special_tokens=self.config.additional_special_tokens,
        )

        self.max_word_length = self.config.max_word_length or 100
        self.bos_token_id = self.config.bos_token_id or 101
        self.eos_token_id = self.config.eos_token_id or 102

    def batch_encode_sentence(self, sentences, labels=None, inplace=True):
        if labels is None:
            labels = [None] * len(sentences)
        assert len(labels) == len(
            sentences
        ), "make sure that sentences and labels have the same length and that they map to each other"
        self.orig_sentences = sentences
        encoded_sentences = [
            self.encode_sentence(s, l) for s, l in zip(sentences, labels)
        ]
        self.input_ids, self.word_begin_indexes, self.sentences, self.labels = zip(
            *encoded_sentences
        )
        if not inplace:
            return input_ids, word_begin_indexes, sentences, labels

    def encode_sentence(self, *args, **kwargs):
        NotImplemented

    def to_allennlp_format(extractions, sentence):
        for extra in extractions:
            args1 = extra.args[0]
            pred = extra.pred
            args2 = " ".join(extra.args[1:])
            confidence = extra.confidence

            extra_sent = f"{sentence}\t<arg1> {args1} </arg1> <rel> {pred} </rel> <arg2> {arg2} </arg2>\t{confidence}"

    def to_dataloader(
        self,
        input_ids=None,
        word_begin_indexes=None,
        sentences=None,
        labels=None,
        batch_size=8,
        shuffle=False,
        collate_fn=None,
    ):
        input_ids = input_ids or self.input_ids
        word_begin_indexes = word_begin_indexes or self.word_begin_indexes
        sentences = sentences or self.sentences
        labels = labels or self.labels

        collate_fn = collate_fn or collate_pad_data(self.config.max_depth)
        self.sentences = sentences
        f_names = self.field_names
        examples = [
            dict(zip(f_names, e))
            for e in zip(input_ids, word_begin_indexes, labels, range(len(sentences)))
        ]
        dataloader = DataLoader(
            examples, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
        )
        return dataloader

    def normalize(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.strip(" |\n")
            sentence = sentence.replace("’", "'")
            sentence = sentence.replace("”", "''")
            sentence = sentence.replace("“", "''")
        return sentence

    @classmethod
    def load_pretrained(cls, path):
        path_file = get_config(path)
        with open(path_file) as f:
            config = json.load(f)
        return cls(config)

    def save_pretrained(self, path):
        config = dict(self.config)
        config.pop("__class__")
        with open(path, "w") as f:
            json.dump(config, f)


class DataForTriplet(Data):
    def __init__(self, config):
        config["max_depth"] = 5
        super().__init__(config)

        self.field_names = ("input_ids", "word_begin_index", "labels", "sentence_index")
        self.label_dict = {
            "NONE": 0,
            "ARG1": 1,
            "REL": 2,
            "ARG2": 3,
            "LOC": 4,
            "TIME": 4,
            "TYPE": 5,
            "ARGS": 3,
        }

    def encode_sentence(self, sentence, labels=None):
        sentence = self.normalize(sentence)
        tokens = nltk.word_tokenize(sentence)
        sentence = " ".join(tokens[: self.max_word_length])
        u_sentence = (
            sentence.strip() + " " + " ".join(self.config.additional_special_tokens)
        )
        word_tokens = self.tokenizer.batch_encode_plus(
            u_sentence.split(), add_special_tokens=False
        )["input_ids"]

        input_ids, word_begin_index = [], []
        for i in word_tokens:
            word_begin_index.append(len(input_ids) + 1)
            input_ids.extend(i or [100])

        input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
        labels = [labels] if isinstance(labels, str) else labels
        if labels and isinstance(labels, list):
            labels = [
                [self.label_dict[i] for i in label_at_depth.split()]
                for label_at_depth in labels
            ]
            labels = [pad_list(l, len(word_begin_index), 0) for l in labels]
        else:
            labels = [[0] * len(word_begin_index)]
        return input_ids, word_begin_index, sentence, labels

    def batch_decode_prediction(self, predictions, scores, sentences=None):
        sentences = sentences or self.sentences
        return [
            DataForTriplet.decode_prediction(pred, score, sent)
            for pred, score, sent in zip(predictions, scores, sentences)
        ]

    @staticmethod
    def decode_prediction(predictions, scores, sentence):
        # prediction: shape of (depth, max_word_length)
        # scores: shape of (depth)

        words = sentence.split() + ADDITIONAL_SPECIAL_TOKENS
        predictions, indices = torch.unique(
            predictions, dim=0, sorted=False, return_inverse=True
        )
        scores = scores[indices.unique()]

        mask_non_null = predictions.sum(dim=-1) != 0
        predictions = predictions[mask_non_null]
        scores = scores[mask_non_null]

        extractions = []
        for prediction, score in zip(predictions, scores):
            prediction = prediction[: len(words)]
            pro_extraction = process_extraction(prediction, words, score.item())
            if pro_extraction.args[0] and pro_extraction.pred:
                extracted_triplet = ext_to_triplet(pro_extraction)
                extractions.append(extracted_triplet)

        output = dict()
        output["triplet"] = extractions
        output["sentence"] = sentence

        return output


class DataForConjunction(Data):
    def __init__(self, config):
        config["max_depth"] = 3
        super().__init__(config)

        self.field_names = ("input_ids", "word_begin_index", "labels", "sentence_index")
        self.label_dict = {
            "CP_START": 2,
            "CP": 1,
            "CC": 3,
            "SEP": 4,
            "OTHERS": 5,
            "NONE": 0,
        }

    def encode_sentence(self, sentence, labels=None):
        sentence = self.normalize(sentence)
        tokens = nltk.word_tokenize(sentence)
        sentence = " ".join(tokens[: self.max_word_length])
        u_sentence = sentence.strip() + " [unused1] [unused2] [unused3]"

        word_tokens = self.tokenizer.batch_encode_plus(
            u_sentence.split(), add_special_tokens=False
        )["input_ids"]
        input_ids, word_begin_index = [], []
        for i in word_tokens:
            word_begin_index.append(len(input_ids) + 1)
            input_ids.extend(i or [100])

        input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
        labels = [labels] if isinstance(labels, str) else labels
        if labels and isinstance(labels, list):

            labels = [
                [self.label_dict[i] for i in label_at_depth.split()]
                for label_at_depth in labels
            ]
            labels = [pad_list(l, len(word_begin_index), 0) for l in labels]
        else:
            labels = [[0] * len(word_begin_index)]
        return input_ids, word_begin_index, sentence, labels

    def batch_decode_prediction(self, predictions, sentences=None, **kw):
        # prediction batch_size, depth, num_words, sentences
        sentences = sentences or self.sentences
        output = []
        for prediction, sentence in zip(predictions, sentences):
            words = sentence.split()
            len_words = len(words)
            prediction = [p[:len_words] for p in prediction.tolist()]
            coords = get_coords(prediction)
            output_sentences = coords_to_sentences(coords, words)
            output.append(
                {
                    "sentence": sentence,
                    "prediction": output_sentences[0],
                    "conjugation_words": output_sentences[1],
                }
            )
        return output


def get_coords(predictions):
    all_coordination_phrases = dict()
    for depth, depth_prediction in enumerate(predictions):
        coordination_phrase, start_index = None, -1
        coordphrase, conjunction, coordinator, separator = False, False, False, False
        for i, label in enumerate(depth_prediction):
            if label != 1:  # conjunction can end
                if conjunction and coordination_phrase != None:
                    conjunction = False
                    coordination_phrase["conjuncts"].append((start_index, i - 1))
            if label == 0 or label == 2:  # coordination phrase can end
                if (
                    coordination_phrase != None
                    and len(coordination_phrase["conjuncts"]) >= 2
                    and coordination_phrase["cc"]
                    > coordination_phrase["conjuncts"][0][1]
                    and coordination_phrase["cc"]
                    < coordination_phrase["conjuncts"][-1][0]
                ):
                    coordination = Coordination(
                        coordination_phrase["cc"],
                        coordination_phrase["conjuncts"],
                        label=depth,
                    )
                    all_coordination_phrases[coordination_phrase["cc"]] = coordination
                    coordination_phrase = None
            if label == 0:
                continue
            if label == 1:  # can start a conjunction
                if not conjunction:
                    conjunction = True
                    start_index = i
            if label == 2:  # starts a coordination phrase
                coordination_phrase = {"cc": -1, "conjuncts": [], "seps": []}
                conjunction = True
                start_index = i
            if label == 3 and coordination_phrase != None:
                coordination_phrase["cc"] = i
            if label == 4 and coordination_phrase != None:
                coordination_phrase["seps"].append(i)
            if label == 5:  # nothing to be done
                continue
            if label == 3 and coordination_phrase == None:
                # coordinating words which do not have associated conjuncts
                all_coordination_phrases[i] = None
    return all_coordination_phrases
