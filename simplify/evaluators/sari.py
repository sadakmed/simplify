from collections import Counter

import numpy as np

NGRAM = 4


def compute_f1(precision, recall):
    return (
        2 * precision * recall / (precision + recall) if precision * recall != 0 else 0
    )


def computer_precision(cs, s):
    return cs / s if s > 0 else 0


def computer_recall(cs, r):
    return cs / r if r > 0 else 0


def keep_precision_recall(original_ngram, sys_ngram, ref_ngram, num_ref):
    o_ngram_counter = Counter(original_ngram * num_ref)
    sys_ngram_counter = Counter(sys_ngram * num_ref)
    ref_ngram_counter = Counter(ref_ngram)

    kept_by_sys = sys_ngram_counter & o_ngram_counter
    kept_correctly_by_sys = kept_by_sys & ref_ngram_counter
    kept_by_ref = o_ngram_counter & ref_ngram_counter

    precision, recall = 0, 0
    for token, val in kept_correctly_by_sys.items():
        precision += val / kept_by_sys[token]
        recall += val / kept_by_ref[token]
    precision = precision / len(kept_by_sys) if kept_by_sys.values() else 0
    recall = recall / len(kept_by_ref) if kept_by_ref.values() else 0

    return precision, recall


def add_precision_recall(original_ngram, sys_ngram, ref_ngram):
    o_ngram_counter = Counter(original_ngram)
    sys_ngram_counter = Counter(sys_ngram)
    ref_ngram_counter = Counter(ref_ngram)

    added_by_sys = set(sys_ngram_counter) - set(o_ngram_counter)
    added_by_ref = set(ref_ngram_counter) - set(o_ngram_counter)
    added_correctly_by_sys = set(added_by_sys) & set(ref_ngram_counter)

    add_precision = computer_precision(len(added_correctly_by_sys), len(added_by_sys))
    add_recall = computer_recall(len(added_correctly_by_sys), len(added_by_ref))
    return add_precision, add_recall


def delete_precision_recall(original_ngram, sys_ngram, ref_ngram, num_ref):
    o_ngram_counter = Counter(original_ngram * num_ref)
    sys_ngram_counter = Counter(sys_ngram * num_ref)
    ref_ngram_counter = Counter(ref_ngram)

    deleted_by_sys = o_ngram_counter - sys_ngram_counter
    deleted_by_ref = o_ngram_counter - ref_ngram_counter
    deleted_correctly_by_sys = deleted_by_sys & deleted_by_ref
    precision, recall = 0, 0
    for token, value in deleted_correctly_by_sys.items():
        precision += value / deleted_by_sys[token]
        recall += value / deleted_by_ref[token]
    precision = precision / len(deleted_by_sys) if deleted_by_sys.values() else 0
    recall = recall / len(deleted_by_ref) if deleted_by_ref.values() else 0

    return precision, recall


def grams(sentence, g=2):
    if g == 1:
        return sentence.lower().split()
    tokens = np.array(sentence.lower().split())
    slices = [slice(i, i + g) for i in range(len(tokens) - g + 1)]
    return [" ".join(tokens[s]) for s in slices]


def sari(original_sentence: str, system_sentence: str, refs: list):

    num_ref = len(refs)
    # should be word tokenized
    action_f1 = []
    for n in range(1, NGRAM + 1):
        original_ngram = grams(original_sentence, g=n)
        sys_ngram = grams(system_sentence, g=n)
        refs_ngram = [token for ref in refs for token in grams(ref, g=n)]

        del_scores = delete_precision_recall(
            original_ngram, sys_ngram, refs_ngram, num_ref
        )
        keep_scores = keep_precision_recall(
            original_ngram, sys_ngram, refs_ngram, num_ref
        )
        add_scores = add_precision_recall(original_ngram, sys_ngram, refs_ngram)

        action_f1.append(del_scores[0])
        action_f1.append(compute_f1(*keep_scores))
        action_f1.append(compute_f1(*add_scores))

    sari_score = sum(action_f1) / (n * 3)
    return sari_score


def main():
    ssent = "About 95 species are currently accepted ."
    csent1 = "About 95 you now get in ."
    csent2 = "About 95 species are now agreed ."
    csent3 = "About 95 species are currently agreed ."
    rsents = [
        "About 95 species are currently known .",
        "About 95 species are now accepted .",
        "95 species are now accepted .",
    ]

    print(SARIsent(ssent, csent1, rsents))
    print(SARIsent(ssent, csent2, rsents))
    print(SARIsent(ssent, csent3, rsents))


if __name__ == "__main__":
    main()
