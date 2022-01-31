from .levenshtein import lev_ratio
from .sari import sari

__all__ = ["sari", "lev_ratio", "compression_ratio"]


def compression_ratio(hypothesis, reference):
    assert isinstance(hypothesis, str) and isinstance(
        reference, str
    ), "input must be in type str"
    if reference:
        return len(hypothesis) / len(reference)
    raise "reference is empty string"


def addition_ratio(hypothesis, reference):
    pass


def deletion_ratio(hypothesis, reference):
    pass
