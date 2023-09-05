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
"""Lifted Structured Loss
    Deep Metric Learning via Lifted Structured Feature Embedding.
    https://arxiv.org/abs/1511.06452
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from tensorflow_similarity.types import FloatTensor, IntTensor
    from tensorflow_similarity.distances import Distance

import tensorflow_similarity.distances
from tensorflow_similarity import losses as tfsim_losses
from tensorflow_similarity.algebra import build_masks

from .metric_loss import MetricLoss
from .utils import positive_distances


def lifted_struct_loss(
    labels: IntTensor,
    embeddings: FloatTensor,
    key_labels: IntTensor,
    key_embeddings: FloatTensor,
    distance: Distance,
    positive_mining_strategy: str = "hard",
    margin: float = 1.0,
) -> FloatTensor:
    """Lifted Struct loss computations"""

    # Compute pairwise distances
    pairwise_distances = distance(embeddings, key_embeddings)

    # Build masks for positive and negative pairs
    positive_mask, negative_mask = build_masks(
        query_labels=labels, key_labels=key_labels, batch_size=tf.shape(embeddings)[0]
    )

    # Get positive distances and indices
    positive_dists, positive_indices = positive_distances(positive_mining_strategy, pairwise_distances, positive_mask)

    # Reorder pairwise distances and negative mask based on positive indices
    reordered_pairwise_distances = tf.gather(pairwise_distances, positive_indices, axis=1)
    reordered_negative_mask = tf.gather(negative_mask, positive_indices, axis=1)

    # Concatenate pairwise distances and negative masks along axis=1
    concatenated_distances = tf.concat([pairwise_distances, reordered_pairwise_distances], axis=1)
    concatenated_negative_mask = tf.concat([negative_mask, reordered_negative_mask], axis=1)
    concatenated_negative_mask = tf.cast(concatenated_negative_mask, tf.float32)
    # Compute (margin - neg_dist) logsum_exp values for each row (equation 4 in the paper)
    neg_logsumexp = tfsim_losses.utils.logsumexp(margin - concatenated_distances, concatenated_negative_mask)

    # Calculate the loss
    j_values = neg_logsumexp + positive_dists

    loss: FloatTensor = j_values / 2.0

    return loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class LiftedStructLoss(MetricLoss):
    """Computes the lifted structured loss in an online fashion.
    This loss encourages the positive distances between a pair of embeddings
    with the same labels to be smaller than the negative distances between pair
    of embeddings of different labels.
    See: https://arxiv.org/abs/1511.06452 for the original paper.
    `y_true` must be a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer values**.
    `y_pred` must be 2-D float `Tensor` of L2 normalized embedding vectors.
    You can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly normalized.
    """

    def __init__(
        self,
        distance: Distance | str = "cosine",
        positive_mining_strategy: str = "hard",
        margin: float = 1.0,
        name: str = "lifted_struct_loss",
        **kwargs,
    ):
        """Initializes the LiftedStructLoss.
        Args:
            distance: Which distance function to use to compute the pairwise
                distances between embeddings.
            positive_mining_strategy: What mining strategy to use to select
                embedding from the same class. Defaults to 'hard'.
                Available: {'easy', 'hard'}
            margin: Use an explicit value for the margin term.
            name: Optional name for the instance. Defaults to 'lifted_struct_loss'.
        Raises:
            ValueError: Invalid positive mining strategy.
        """

        # distance canonicalization
        self.distance = tensorflow_similarity.distances.get(distance)

        # sanity checks
        if positive_mining_strategy not in ["easy", "hard"]:
            raise ValueError("Invalid positive mining strategy")

        super().__init__(
            lifted_struct_loss,
            name=name,
            # The following are passed to the lifted_struct_loss function as fn_kwargs
            distance=self.distance,
            positive_mining_strategy=positive_mining_strategy,
            margin=margin,
            **kwargs,
        )
