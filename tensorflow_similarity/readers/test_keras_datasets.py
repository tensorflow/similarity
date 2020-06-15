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

from tensorflow_similarity.readers.keras_datasets import KerasDatasetReader


def test_cifar10():
    train = KerasDatasetReader(dataset='cifar10', split='train').read()
    test = KerasDatasetReader(dataset='cifar10', split='test').read()

    assert 'examples' in train
    assert 'labels' in train

    assert 50000 == len(train['examples'])
    assert 50000 == len(train['labels'])

    assert 'examples' in test
    assert 'labels' in test
    assert 'groups' in test

    assert 10000 == len(test['examples'])
    assert 10000 == len(test['labels'])
    assert 10000 == len(test['groups'])

    assert test['groups'][0] == 'targets'
