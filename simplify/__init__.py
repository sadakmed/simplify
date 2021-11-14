import os
from pathlib import Path

from .evaluators import *

HOME = Path(os.environ["HOME"])
SIMPLIFY_CACHE = HOME / ".cache/simplify"

if not os.path.exists(SIMPLIFY_CACHE):
    os.mkdir(SIMPLIFY_CACHE)
