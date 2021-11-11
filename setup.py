import os
import re
import shutil
from distutils.core import Command, setup
from pathlib import Path

from setuptools import find_packages
packages = ["simplify."+p for p in find_packages("simplify")]
setup(name="simplify", version="0.0.0", packages=packages + ["simplify"])
