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
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.7.11"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")

setup(
    name="tensorflow_similarity_alpha",
    version=get_version("tensorflow_similarity/__init__.py"),
    description="Single shot and metric learning for humans",
    long_description=long_description,
    author='Tensorflow Similarity authors',
    author_email='tensorflow_similarity@google.com',
    url='https://github.com/tensorflow/similarity',
    license='Apache License 2.0',
    install_requires=[
        'numpy', 'tabulate', 'nmslib', 'tensorflow>=2.2.0', 'tqdm',
        'matplotlib', 'pyarrow'
    ],
    extras_require={
        "dev": [
            'pytest', 'pytest-cov', 'flake8', 'twine', 'perfCounters',
            'pytype'
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console', 'Framework :: Jupyter',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    package_dir={"": "tensorflow_similarity"},
    packages=find_packages(where="tensorflow_similarity")
    )