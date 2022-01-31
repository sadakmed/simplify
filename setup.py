import shutil
from pathlib import Path
from setuptools import setup, find_packages

from simplify import SIMPLIFY_CACHE

shutil.copy(Path(".jar/discourse.jar"), SIMPLIFY_CACHE)


packages = ['simplify']
setup(name="simplify", version="0.0.0", packages=packages)
