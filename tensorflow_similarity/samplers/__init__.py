# Copyright 2021 The TensorFlow Authors
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

"""Data Samplers generate balanced batches for smooth training.

*A well balanced batch is a batch that contains at least 2 examples for
each class present in the batch*.

Having well balanced batches is important for many types of similarity learning
including contrastive learning because contrastive losses require at least
two examples (and sometimes more) to be able to compute distances between
the embeddings.

To address this need, TensorFlow Similarity provides data samplers for
various types of datasets that:
- Ensure that batches contain at least N examples of each class present in
the batch.
- Support restricting the batches to a subset of the classes present in
the dataset.
"""
from .utils import select_examples  # noqa
from .memory_samplers import MultiShotMemorySampler  # noqa
from .memory_samplers import SingleShotMemorySampler  # noqa
from .tfrecords_samplers import TFRecordDatasetSampler  # noqa
from .tfdataset_samplers import TFDatasetMultiShotMemorySampler  # noqa
