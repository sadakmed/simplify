# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil

from .preprocessors import get_preprocessors
from .utils import write_lines, read_lines, get_temp_filepath, download_and_extract
from .simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from simplify import SIMPLIFY_CACHE

__all__ = ["Muss"]

ALLOWED_MODEL_NAMES = [
    "muss_en_wikilarge_mined",
    "muss_en_mined",
]


preprocessors_kwargs = {
    "LengthRatioPreprocessor": {"target_ratio": 0.9, "use_short_name": False},
    "ReplaceOnlyLevenshteinPreprocessor": {
        "target_ratio": 0.65,
        "use_short_name": False,
    },
    "WordRankRatioPreprocessor": {"target_ratio": 0.75, "use_short_name": False},
    "DependencyTreeDepthRatioPreprocessor": {
        "target_ratio": 0.4,
        "use_short_name": False,
    },
    "GPT2BPEPreprocessor": {},
}


def get_model_path(model_name):
    assert model_name in ALLOWED_MODEL_NAMES
    model_path = SIMPLIFY_CACHE / model_name
    if not model_path.exists():
        url = f"https://dl.fbaipublicfiles.com/muss/{model_name}.tar.gz"
        extracted_path = download_and_extract(url)[0]
        shutil.move(extracted_path, model_path)
    return model_path


class Muss:
    def __init__(self, model_name: str = "muss_en_wikilarge_mined"):
        self.model_name = model_name
        self.preprocessors = None
        self.simplifier = None
        self._initialize()

    def _initialize(self):
        model_path = get_model_path(self.model_name)
        self.preprocessors = get_preprocessors(preprocessors_kwargs)
        simplifier = get_fairseq_simplifier(model_path)
        self.simplifier = get_preprocessed_simplifier(simplifier, self.preprocessors)

    def __call__(self, sentences):
        source_path = get_temp_filepath()
        write_lines(sentences, source_path)
        prediction_path = self.simplifier(source_path)
        prediction_sentences = read_lines(prediction_path)
        return prediction_sentences
