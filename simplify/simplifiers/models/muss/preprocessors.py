# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from functools import wraps, lru_cache
import hashlib
from pathlib import Path

# import dill as pickle
import shutil

import re

import numpy as np

from fairseq.data.encoders.gpt2_bpe_utils import get_encoder

from .utils import (
    write_lines_in_parallel,
    yield_lines_in_parallel,
    add_dicts,
    get_default_args,
    get_temp_filepath,
    failsafe_division,
    download,
    download_and_extract,
    yield_lines,
)

from simplify import SIMPLIFY_CACHE
from simplify.evaluators import lev_ratio

FATTEXT_EMBEDDINGS_DIR = SIMPLIFY_CACHE / "fasttext-vectors/"
SPECIAL_TOKEN_REGEX = r"<[a-zA-Z\-_\d\.]+>"


def get_fasttext_embeddings_path(language="en"):
    fasttext_embeddings_path = FASTTEXT_EMBEDDINGS_DIR / f"cc.{language}.300.vec"
    if not fasttext_embeddings_path.exists():
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz"
        fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(download_and_extract(url)[0], fasttext_embeddings_path)
    return fasttext_embeddings_path


@lru_cache(maxsize=10)
def get_spacy_model(language="en", size="md"):
    # Inline lazy import because importing spacy is slow
    import spacy

    if language == "it" and size == "md":
        print(
            "Model it_core_news_md is not available for italian, falling back to it_core_news_sm"
        )
        size = "sm"
    model_name = {
        "en": f"en_core_web_{size}",
        "fr": f"fr_core_news_{size}",
        "es": f"es_core_news_{size}",
        "it": f"it_core_news_{size}",
        "de": f"de_core_news_{size}",
    }[language]
    return spacy.load(model_name)  # python -m spacy download en_core_web_sm


@lru_cache(maxsize=10 ** 6)
def spacy_process(text, language="en", size="md"):
    return get_spacy_model(language=language, size=size)(str(text))


@lru_cache(maxsize=1)
def get_spacy_tokenizer(language="en"):
    return get_spacy_model(language=language).Defaults.create_tokenizer(
        get_spacy_model(language=language)
    )


def get_spacy_content_tokens(text, language="en"):
    def is_content_token(token):
        return (
            not token.is_stop and not token.is_punct and token.ent_type_ == ""
        )  # Not named entity

    return [
        token
        for token in get_spacy_tokenizer(language=language)(text)
        if is_content_token(token)
    ]


def get_content_words(text, language="en"):
    return [token.text for token in get_spacy_content_tokens(text, language=language)]


@lru_cache(maxsize=10)
def get_word2rank(vocab_size=10 ** 5, language="en"):
    word2rank = {}
    line_generator = yield_lines(get_fasttext_embeddings_path(language))
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(" ")[0]
        word2rank[word] = i
    return word2rank


def get_rank(word, language="en"):
    return get_word2rank(language=language).get(
        word, len(get_word2rank(language=language))
    )


def get_log_rank(word, language="en"):
    return np.log(1 + get_rank(word, language=language))


def get_log_ranks(text, language="en"):
    return [
        get_log_rank(word, language=language)
        for word in get_content_words(text, language=language)
        if word in get_word2rank(language=language)
    ]


# Single sentence feature extractors with signature function(sentence) -> float
def get_lexical_complexity_score(sentence, language="en"):
    log_ranks = get_log_ranks(sentence, language=language)
    if len(log_ranks) == 0:
        log_ranks = [
            np.log(1 + len(get_word2rank()))
        ]  # TODO: This is completely arbitrary
    return np.quantile(log_ranks, 0.75)


def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return lev_ratio(complex_sentence, simple_sentence)


def get_levenshtein_distance(complex_sentence, simple_sentence):
    # We should rename this to get_levenshtein_distance_ratio for more clarity
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence)


def get_replace_only_levenshtein_distance(complex_sentence, simple_sentence):
    return len(
        [
            _
            for operation, _, _ in Levenshtein.editops(
                complex_sentence, simple_sentence
            )
            if operation == "replace"
        ]
    )


def get_replace_only_levenshtein_distance_ratio(complex_sentence, simple_sentence):
    max_replace_only_distance = min(len(complex_sentence), len(simple_sentence))
    return failsafe_division(
        get_replace_only_levenshtein_distance(complex_sentence, simple_sentence),
        max_replace_only_distance,
        default=0,
    )


def get_replace_only_levenshtein_similarity(complex_sentence, simple_sentence):
    return 1 - get_replace_only_levenshtein_distance_ratio(
        complex_sentence, simple_sentence
    )


def get_dependency_tree_depth(sentence, language="en"):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [
        get_subtree_depth(spacy_sentence.root)
        for spacy_sentence in spacy_process(sentence, language=language).sents
    ]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)


PREPROCESSORS_REGISTRY = {}


def get_preprocessor_by_name(preprocessor_name):
    return PREPROCESSORS_REGISTRY[preprocessor_name]


def get_preprocessors(preprocessors_kwargs):
    preprocessors = []
    for preprocessor_name, kwargs in preprocessors_kwargs.items():
        preprocessors.append(get_preprocessor_by_name(preprocessor_name)(**kwargs))
    return preprocessors


def extract_special_tokens(sentence):
    """Remove any number of token at the beginning of the sentence"""
    match = re.match(fr"(^(?:{SPECIAL_TOKEN_REGEX} *)+) *(.*)$", sentence)
    if match is None:
        return "", sentence
    special_tokens, sentence = match.groups()
    return special_tokens.strip(), sentence


def remove_special_tokens(sentence):
    return extract_special_tokens(sentence)[1]


def store_args(constructor):
    @wraps(constructor)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, "args") or not hasattr(self, "kwargs"):
            # TODO: Default args are not overwritten if provided as args
            self.args = args
            self.kwargs = add_dicts(get_default_args(constructor), kwargs)
        return constructor(self, *args, **kwargs)

    return wrapped


# def dump_preprocessors(preprocessors, dir_path):
#     with open(Path(dir_path) / 'preprocessors.pickle', 'wb') as f:
#         pickle.dump(preprocessors, f)


# def load_preprocessors(dir_path):
#     path = Path(dir_path) / 'preprocessors.pickle'
#     if not path.exists():
#         return None
#     with open(path, 'rb') as f:
#         return pickle.load(f)


class AbstractPreprocessor(ABC):
    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        PREPROCESSORS_REGISTRY[cls.__name__] = cls

    def __repr__(self):
        args = getattr(self, "args", ())
        kwargs = getattr(self, "kwargs", {})
        args_repr = [repr(arg) for arg in args]
        kwargs_repr = [
            f"{k}={repr(v)}" for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])
        ]
        args_kwargs_str = ", ".join(args_repr + kwargs_repr)
        return f"{self.__class__.__name__}({args_kwargs_str})"

    def get_hash_string(self):
        return self.__class__.__name__

    def get_hash(self):
        return hashlib.md5(self.get_hash_string().encode()).hexdigest()

    @staticmethod
    def get_nevergrad_variables():
        return {}

    @property
    def prefix(self):
        return self.__class__.__name__.replace("Preprocessor", "")

    def fit(self, complex_filepath, simple_filepath):
        pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def decode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        if complex_sentence is not None:
            complex_sentence = self.encode_sentence(complex_sentence)
        if simple_sentence is not None:
            simple_sentence = self.encode_sentence(simple_sentence)
        return complex_sentence, simple_sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            # We will use an empty temporary file which will yield None for each line
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            for input_line, encoder_line in yield_lines_in_parallel(
                [input_filepath, encoder_filepath], strict=False
            ):
                f.write(self.encode_sentence(input_line, encoder_line) + "\n")

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            # We will use an empty temporary file which will yield None for each line
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            for encoder_sentence, input_sentence in yield_lines_in_parallel(
                [encoder_filepath, input_filepath], strict=False
            ):
                decoded_sentence = self.decode_sentence(
                    input_sentence, encoder_sentence=encoder_sentence
                )
                f.write(decoded_sentence + "\n")

    def encode_file_pair(
        self,
        complex_filepath,
        simple_filepath,
        output_complex_filepath,
        output_simple_filepath,
    ):
        """Jointly encode a complex file and a simple file (can be aligned or not)"""
        with write_lines_in_parallel(
            [output_complex_filepath, output_simple_filepath], strict=False
        ) as output_files:
            for complex_line, simple_line in yield_lines_in_parallel(
                [complex_filepath, simple_filepath], strict=False
            ):
                output_files.write(self.encode_sentence_pair(complex_line, simple_line))


class ComposedPreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(self, preprocessors, sort=False):
        if preprocessors is None:
            preprocessors = []
        if sort:
            # Make sure preprocessors are always in the same order
            preprocessors = sorted(
                preprocessors, key=lambda preprocessor: preprocessor.__class__.__name__
            )
        self.preprocessors = preprocessors

    def get_hash_string(self):
        preprocessors_hash_strings = [
            preprocessor.get_hash_string() for preprocessor in self.preprocessors
        ]
        return f"ComposedPreprocessor(preprocessors={preprocessors_hash_strings})"

    def get_suffix(self):
        return "_".join([p.prefix.lower() for p in self.preprocessors])

    def fit(self, complex_filepath, simple_filepath):
        for preprocessor in self.preprocessors:
            pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.encode_sentence(sentence, encoder_sentence)
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.decode_sentence(sentence, encoder_sentence)
        return sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        for preprocessor in self.preprocessors:
            intermediary_output_filepath = get_temp_filepath()
            preprocessor.encode_file(
                input_filepath, intermediary_output_filepath, encoder_filepath
            )
            input_filepath = intermediary_output_filepath
        shutil.copyfile(input_filepath, output_filepath)

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        for preprocessor in self.preprocessors:
            intermediary_output_filepath = get_temp_filepath()
            preprocessor.decode_file(
                input_filepath, intermediary_output_filepath, encoder_filepath
            )
            input_filepath = intermediary_output_filepath
        shutil.copyfile(input_filepath, output_filepath)

    def encode_file_pair(
        self,
        complex_filepath,
        simple_filepath,
        output_complex_filepath,
        output_simple_filepath,
    ):
        for preprocessor in self.preprocessors:
            intermediary_output_complex_filepath = get_temp_filepath()
            intermediary_output_simple_filepath = get_temp_filepath()
            preprocessor.encode_file_pair(
                complex_filepath,
                simple_filepath,
                intermediary_output_complex_filepath,
                intermediary_output_simple_filepath,
            )
            complex_filepath = intermediary_output_complex_filepath
            simple_filepath = intermediary_output_simple_filepath
        shutil.copyfile(complex_filepath, output_complex_filepath)
        shutil.copyfile(simple_filepath, output_simple_filepath)

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        for preprocessor in self.preprocessors:
            complex_sentence, simple_sentence = preprocessor.encode_sentence_pair(
                complex_sentence, simple_sentence
            )
        return complex_sentence, simple_sentence


class FeaturePreprocessor(AbstractPreprocessor):
    """Prepend a computed feature at the beginning of the sentence"""

    @store_args
    def __init__(
        self,
        feature_name,
        get_feature_value,
        get_target_feature_value,
        bucket_size=0.05,
        noise_std=0,
        prepend_to_target=False,
        use_short_name=False,
    ):
        self.get_feature_value = get_feature_value
        self.get_target_feature_value = get_target_feature_value
        self.bucket_size = bucket_size
        self.noise_std = noise_std
        self.feature_name = feature_name.upper()
        self.use_short_name = use_short_name
        if use_short_name:
            # There might be collisions
            self.feature_name = self.feature_name.lower()[:4]
        self.prepend_to_target = prepend_to_target

    def get_hash_string(self):
        return f"{self.__class__.__name__}(feature_name={repr(self.feature_name)}, bucket_size={self.bucket_size}, noise_std={self.noise_std}, prepend_to_target={self.prepend_to_target}, use_short_name={self.use_short_name})"  # noqa: E501

    def bucketize(self, value):
        """Round value to bucket_size to reduce the number of different values"""
        return round(round(value / self.bucket_size) * self.bucket_size, 10)

    def add_noise(self, value):
        return value + np.random.normal(0, self.noise_std)

    def get_feature_token(self, feature_value):
        return f"<{self.feature_name}_{feature_value}>"

    def encode_sentence(self, sentence, encoder_sentence=None):
        if not self.prepend_to_target:
            desired_feature = self.bucketize(
                self.get_target_feature_value(remove_special_tokens(sentence))
            )
            sentence = f"{self.get_feature_token(desired_feature)} {sentence}"
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        if self.prepend_to_target:
            _, sentence = extract_special_tokens(sentence)
        return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        feature = self.bucketize(
            self.add_noise(
                self.get_feature_value(
                    remove_special_tokens(complex_sentence),
                    remove_special_tokens(simple_sentence),
                )
            )
        )
        if self.prepend_to_target:
            simple_sentence = f"{self.get_feature_token(feature)} {simple_sentence}"
        else:
            complex_sentence = f"{self.get_feature_token(feature)} {complex_sentence}"
        return complex_sentence, simple_sentence


class LevenshteinPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(),
            self.get_feature_value,
            self.get_target_feature_value,
            bucket_size,
            noise_std,
            **kwargs,
        )

    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_levenshtein_similarity(complex_sentence, simple_sentence)

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class ReplaceOnlyLevenshteinPreprocessor(LevenshteinPreprocessor):
    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_replace_only_levenshtein_similarity(
            complex_sentence, simple_sentence
        )


class RatioPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(
        self,
        feature_extractor,
        target_ratio=0.8,
        bucket_size=0.05,
        noise_std=0,
        **kwargs,
    ):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(),
            self.get_feature_value,
            self.get_target_feature_value,
            bucket_size,
            noise_std,
            **kwargs,
        )

    def get_feature_value(self, complex_sentence, simple_sentence):
        return min(
            failsafe_division(
                self.feature_extractor(simple_sentence),
                self.feature_extractor(complex_sentence),
            ),
            2,
        )

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class LengthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(len, *args, **kwargs)


class WordRankRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language="en", **kwargs):
        super().__init__(
            lambda sentence: get_lexical_complexity_score(sentence, language=language),
            *args,
            **kwargs,
        )


class DependencyTreeDepthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language="en", **kwargs):
        super().__init__(
            lambda sentence: get_dependency_tree_depth(sentence, language=language),
            *args,
            **kwargs,
        )


class GPT2BPEPreprocessor(AbstractPreprocessor):
    def __init__(self):
        self.bpe_dir = SIMPLIFY_CACHE / "bart_bpe"
        self.bpe_dir.mkdir(exist_ok=True, parents=True)
        self.encoder_json_path = self.bpe_dir / "encoder.json"
        self.vocab_bpe_path = self.bpe_dir / "vocab.bpe"
        self.dict_path = self.bpe_dir / "dict.txt"
        download(
            "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json",
            self.encoder_json_path,
            overwrite=False,
        )
        download(
            "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe",
            self.vocab_bpe_path,
            overwrite=False,
        )
        download(
            "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt",
            self.dict_path,
            overwrite=False,
        )

    @property
    @lru_cache(maxsize=1)
    def bpe_encoder(self):
        """
        We need to use a property because GPT2BPEPreprocessor() is cannot pickled
        > pickle.dumps(GPT2BPEPreprocessor())
        ----> TypeError: can't pickle module objects
        """
        return get_encoder(self.encoder_json_path, self.vocab_bpe_path)

    def encode_sentence(self, sentence, *args, **kwargs):
        return " ".join([str(idx) for idx in self.bpe_encoder.encode(sentence)])

    def decode_sentence(self, sentence, *args, **kwargs):
        return self.bpe_encoder.decode([int(idx) for idx in sentence.split(" ")])
