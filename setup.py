import os
import re
import shutil
from distutils.core import setup
from pathlib import Path

from setuptools import find_packages
from simplify import SIMPLIFY_CACHE



shutil.copy(Path(".jar/discourse.jar"), SIMPLIFY_CACHE)

setup(
    name="simplify",
    version="0.0.0",
    packages=["simplify", "simplify.simplifiers", "simplify.evaluators"],
)
