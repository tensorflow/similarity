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
import tensorflow as tf

from tensorflow_similarity.types import IntTensor
from tensorflow_similarity.types import BoolTensor


def compute_match_mask(query_labels: IntTensor,
                       lookup_labels: IntTensor) -> BoolTensor:
    """Compute a boolean mask (indicator function) marking the TPs in the results.

    Args:
        query_labels: A 1D tensor of the labels associated with the queries.

        lookup_labels: A 2D tensor where the jth row is the labels associated
        with the set of k neighbors for the jth query.

    Returns:
        A 2D boolean tensor indicating which lookups match the label of their
        associated query.
    """
    if tf.rank(query_labels) == 1:
        query_labels = tf.expand_dims(query_labels, axis=-1)

    match_mask: BoolTensor = tf.math.equal(lookup_labels, query_labels)

    return match_mask
