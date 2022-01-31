from ..base import _BaseModel
from multiprocessing import Pool
import json


__all__ = ["DiscourseSimplification"]


jarpath = "https://github.com/sadakmed/simplify/raw/master/.jar/discourse.jar"


def with_jvm(paths):
    def nested_decorator(func):
        global wrapper

        def wrapper(sentences):
            import jpype
            import jpype.imports

            jpype.startJVM("-ea", classpath=paths)
            from org.slf4j.Logger import ROOT_LOGGER_NAME
            from org.lambda3.text.simplification.discourse.processing import (
                DiscourseSimplifier,
                ProcessingType,
            )

            logging = jpype.java.util.logging
            off = logging.Level.OFF
            logging.Logger.getLogger(ROOT_LOGGER_NAME).setLevel(off)

            modules = {
                "jpype": jpype,
                "DiscourseSimplifier": DiscourseSimplifier,
                "ProcessingType": ProcessingType,
            }
            func.__globals__.update(modules)

            simple_sentences = func(sentences)

            jpype.shutdownJVM()
            return simple_sentences

        return wrapper

    return nested_decorator


@with_jvm(jarpath)
def discourse_simplify(sentences: list):
    jlist_sentences = jpype.java.util.ArrayList(sentences)
    dis = DiscourseSimplifier()
    j_simple_sentences = dis.doDiscourseSimplification(
        jlist_sentences, ProcessingType.SEPARATE
    )
    p_simple_sentences = str(j_simple_sentences.serializeToJSON().toString())
    simple_sentences = json.loads(p_simple_sentences)
    return simple_sentences


class DiscourseSimplification(_BaseModel):
    def __init__(self):
        pass

    def simplify(self, input_sentences):
        with Pool(1) as p:
            outputs = p.map(discourse_simplify, [input_sentences])
        return outputs





