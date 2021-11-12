import numpy as np
import sacrebleu


class BLEU:
    def __init__(
        self, smooth_method="exp", smooth_value=None, lowercase=False, tokenizer="13a"
    ):
        self.smooth_method = smooth_method
        self.smooth_value = smooth_value
        self.lowercase = lowercase
        self.tokenizer = tokenizer

    def compute(self, system_sentence, refs):
        return sacrebleu.sentence_bleu(
            hypothesis_sentence,
            refs,
            self.smooth_method,
            self.smooth_value,
            self.lowercase,
            self.tokenizer,
        )

    def compute_batch(self, hypothesis_sentences: List[str], refs: List[List[str]]):
        assert len(hypothesis_sentences) == len(
            refs
        ), "number of hypothesis sentences should match the reference sentences"
        return [
            self.compute(hypo, ref) for hypo, ref in zip(hypothesis_sentences, refs)
        ]
