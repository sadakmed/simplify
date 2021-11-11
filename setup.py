import os
import re
import shutil
from distutils.core import Command, setup
from pathlib import Path

from setuptools import find_packages
packages = ["simplify."+p for p in find_packages("simplify")]
print(packages)
setup(name="simplify", version="0.0.0", packages=["simplify", "simplify.simplifiers",
                                                  "simplify.evaluators"])
