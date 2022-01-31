import shutil
import os
from pathlib import Path
from setuptools import setup


SIMPLIFY_CACHE = os.path.expanduser(os.path.join("~/.cache", "simplify"))
if not os.path.exists(SIMPLIFY_CACHE):
    os.mkdir(SIMPLIFY_CACHE)
shutil.copy(Path(".jar/discourse.jar"), SIMPLIFY_CACHE)


packages = ["simplify"]
setup(name="simplify", version="0.0.0", packages=packages)
