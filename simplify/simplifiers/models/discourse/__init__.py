"""
Lambda-3/DiscourseSimplification is licensed under the

GNU General Public License v3.0
Permissions of this strong copyleft license are conditioned on making available complete source code of licensed works and modifications, which include larger works using a licensed work, under the same license. Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
"""


import json
from multiprocessing import Pool

from simplify import SIMPLIFY_CACHE

__all__ = ["Discourse"]
jarpath = SIMPLIFY_CACHE / "discourse.jar"


def with_jvm(paths):
    def nested_decorator(func):
        global wrapper

        def wrapper(sentences):
            import jpype
            import jpype.imports

            jpype.startJVM("-ea", classpath=paths)
            from org.lambda3.text.simplification.discourse.processing import (
                DiscourseSimplifier,
                ProcessingType,
            )
            from org.slf4j.Logger import ROOT_LOGGER_NAME

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
def discourse(sentences: list):
    jlist_sentences = jpype.java.util.ArrayList(sentences)
    dis = DiscourseSimplifier()
    j_simple_sentences = dis.doDiscourseSimplification(
        jlist_sentences, ProcessingType.SEPARATE
    )
    p_simple_sentences = str(j_simple_sentences.serializeToJSON().toString())
    simple_sentences = json.loads(p_simple_sentences)
    return simple_sentences


class Discourse:
    def __init__(self):
        pass

    def __call__(self, sentences):
        with Pool(1) as p:
            # output = p.map(partial(discourse, paths=jarpath), [sentences])
            output = p.map(discourse, [sentences])
        return output
