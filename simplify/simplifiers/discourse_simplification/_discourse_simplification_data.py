import logging

from simplify.simplifiers.base import _BaseData

__all__ = ["DiscourseSimplificationData"]


class DiscourseSimplificationData(_BaseData):
    def __init__(self):
        ...

    def encode(self, sentence):
        return sentence

    def decode(self, output):
        logging.warning("The output is not processed!")
        return output
