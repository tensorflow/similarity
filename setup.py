# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# We use the same setup.py for both tensorflow_similarity and tfsim-nightly
# packages. The package is controlled from the argument line when building the
# pip package.
project_name = "tensorflow_similarity"
if "--project_name" in sys.argv:
    project_name_idx = sys.argv.index("--project_name")
    project_name = sys.argv[project_name_idx + 1]
    sys.argv.remove("--project_name")
    sys.argv.pop(project_name_idx)


setup(
    name=project_name,
    version=get_version("tensorflow_similarity/__init__.py"),
    description="Metric Learning for Humans",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Tensorflow Similarity authors",
    author_email="tf-similarity@google.com",
    url="https://github.com/tensorflow/similarity",
    license="Apache License 2.0",
    install_requires=[
        "bokeh",
        "distinctipy",
        "matplotlib",
        "nmslib",
        "numpy",
        "pandas",
        "Pillow",
        "tabulate",
        "tensorflow>=2.4",
        "tensorflow-datasets>=4.2",
        "tqdm",
        "umap-learn",
    ],
    extras_require={
        "dev": [
            "flake8",
            "mkdocs",
            "mkdocs-autorefs",
            "mkdocs-material",
            "mkdocstrings",
            "mypy",
            "pytest",
            "pytype",
            "setuptools",
            "types-termcolor",
            "twine",
            "types-tabulate",
            "wheel",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
)
