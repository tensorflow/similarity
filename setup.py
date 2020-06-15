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

"""Moirai setup script."""

from setuptools import find_packages
from setuptools import setup

__version__ = '0.1000017'

REQUIREMENTS = """
absl-py
confusable_homoglyphs
flask
h5py
imageio
imgaug
keras-tuner
matplotlib==3.1.0
opencv-python
pillow
scipy
seaborn
sklearn
tabulate
tensorboard
tensorflow-plot
termcolor
tqdm
uuid
""".splitlines()

# Do not require already-installed packages, this avoid replacing system-wide
# installed and optimized packages with local ones.
try:
    import numpy as _
except ImportError:
    REQUIREMENTS.append('numpy')

try:
    import pandas as _
except ImportError:
    REQUIREMENTS.append('pandas')

try:
    import tensorflow as _
except ImportError:
    REQUIREMENTS.append('tensorflow>=2.0.0')

print(find_packages())

setup(
    name='tensorflow-similarity',
    version=__version__,
    description='Tensorflow Similarity: triplet loss and beyond in Keras.',
    author='The tensorflow similarity authors',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
