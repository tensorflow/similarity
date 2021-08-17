# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Multi Label Metric loss (ML2)
    Deep metric learning for multi-labelled radiographs
    https://arxiv.org/pdf/1712.07682
"""

import tensorflow as tf
from typing import Any, Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor
from .utils import negative_distances
from .metric_loss import MetricLoss


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def ml2_loss(labels: IntTensor,
             embeddings: FloatTensor,
             distance: Callable,
             negative_mining_strategy: str = 'semi-hard',
             margin: float = 1.0) -> Any:
    """Multi Label Metric loss (ML2) computations

    This loss takes the jaccard distance between the set of labels associated
    with the anchor and positive examples to weight the margin
    (See section 3.2 in the paper).

    Args:
        labels: A 2D IntTensor of labels associated with the embedded examples.

        embeddings: [description]

        distance: [description]. Defaults to 'cosine'.

        negative_mining_strategy (str, optional): [description].
        Defaults to 'semi-hard'.

        margin: [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    batch_size = tf.size(labels)

    # [distances]
    pairwise_distances = distance(embeddings)

    # Compute the ratio of the intersection of the tags across examples
    labels = tf.cast(labels, dtype=tf.dtypes.bool)

    union = tf.reshape(labels, [batch_size[0], 1, batch_size[1]])
    union = tf.math.logical_or(union, labels),
    union = tf.cast(union, dtype=tf.dtypes.float32)
    union = tf.math.reduce_sum(union, axis=2)

    intersection = tf.reshape(labels, [batch_size[0], 1, batch_size[1]])
    intersection = tf.math.logical_and(intersection, labels)
    intersection = tf.cast(intersection, dtype=tf.float32)
    intersection = tf.math.reduce_sum(intersection, axis=2)

    # We set the floor of the denominator to 1 in order to handle examples that
    # don't have any tags, e.g., [0,0,0,0]. This should make the example a
    # potential negative for all other examples.
    tau = tf.cast(
        (union - intersection) / tf.math.maximum(union, 1),
        dtype=tf.dtypes.float32,
    )

    # Build pairwise binary negative_mask matrix.
    negative_mask = tf.math.equal(tau, 1)
    # Invert so we can select postives only.
    positive_mask = tf.math.logical_not(negative_mask)

    mask_positives = (
        tf.cast(positive_mask, dtype=tf.dtypes.float32) -
        tf.linalg.diag(tf.ones([batch_size[0]])))

    diff = margin - pairwise_distances

    # [Negative distances computation]
    neg_distances, _ = negative_distances(
            negative_mining_strategy,
            diff,
            negative_mask,
            positive_mask,
            batch_size,
    )

    ml2_loss = tf.maximum(
        (pairwise_distances * mask_positives) - margin * tau + neg_distances,
        0.0
    )

    # Get final mean ml2 loss.
    # The denominator is the number of positives, excluding the anchor tag.
    # Again, we clip the floor of the denominator to ensure we don't divide by
    # zero, and we mask out any anchor that has zero positives associated with
    # it.
    mask_positives_1d = tf.cast(
        tf.math.greater(
            tf.math.reduce_sum(mask_positives, axis=1),
            tf.constant([0.0])
        ),
        dtype=tf.dtypes.float32
    )
    ml2_loss = (
        (tf.math.reduce_sum(ml2_loss, axis=1) * mask_positives_1d) /
        tf.math.maximum(tf.math.reduce_sum(mask_positives, axis=1), 1))

    return ml2_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class ML2Loss(MetricLoss):
    """Computes the Multi Label Metric loss in an online fashion.

    This loss takes the jaccard distance between the set of labels associated
    with the anchor and positive examples to weight the margin
    (See section 3.2 in the paper).

    See ML2 loss:
    Deep metric learning for multi-labelled radiographs
    https://arxiv.org/pdf/1712.07682

    `y_true` must be  a 2-D integer `Tensor` of shape (batch_size, num_labels).
    It's values represent all the labels associated with the examples as
    **integer  values** where a 1 associates the label with an example and 0
    otherwise.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """

    def __init__(self,
                 distance: Union[Distance, str] = 'cosine',
                 positive_mining_strategy: str = 'hard',
                 negative_mining_strategy: str = 'hard',
                 soft_margin: bool = False,
                 margin: float = 1.0,
                 name: str = None):
        """Initializes the ML2 Loss

        Args:
            distance (Un, optional): Which distance function to use to compute
            the pairwise distances between embeddings. Defaults to 'cosine'.

            margin (float, optional): Use an explicit value for the margin
            term. Defaults to 1.0.

            name (str, optional): Loss name. Defaults to None.

        Raises:
            ValueError: Invalid positive mining strategy.
            ValueError: Invalid negative mining strategy.
            ValueError: Margin value is not used when soft_margin is set
                        to True.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance
        # sanity checks

        if positive_mining_strategy not in ['easy', 'hard']:
            raise ValueError('Invalid positive mining strategy.')

        if negative_mining_strategy not in ['easy', 'hard', 'semi-hard']:
            raise ValueError('Invalid negative mining strategy.')

        # Ensure users knows its one or the other
        if margin != 1.0 and soft_margin:
            raise ValueError('Margin value is not used when soft_margin is\
                              set to True')

        super().__init__(ml2_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         negative_mining_strategy=negative_mining_strategy,
                         margin=margin)
