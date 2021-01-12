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

from setuptools import find_packages
from setuptools import setup
from time import time

long_description = open("README.md").read()
# version = '0.15.3r%s' % int(time())
version = '0.16.2'

setup(name="tensorflow_similarity_alpha",
      version=version,
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
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console', 'Framework :: Jupyter',
          'License :: OSI Approved :: Apache Software License',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      packages=find_packages())
