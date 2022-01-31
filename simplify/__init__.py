import os
from pathlib import Path




HOME = os.environ["HOME"]
SIMPLIFY_CACHE = os.path.join(HOME , ".cache/simplify")


if not os.path.exists(SIMPLIFY_CACHE):
    os.mkdir(SIMPLIFY_CACHE)
