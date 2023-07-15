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

from typing import Any

import tensorflow as tf

from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor
from tensorflow_similarity import losses as tfsim_losses
from .metric_loss import MetricLoss
from .utils import compute_loss, negative_distances, positive_distances


def lifted_struct_loss(
    labels: IntTensor,
    embeddings: FloatTensor,
    distance: Distance,
    positive_mining_strategy: str = "hard",
    negative_mining_strategy: str = "easy",
    margin: float = 1.0,
) -> FloatTensor:
    """Lifted Struct loss computations"""

    # Compute pairwise distances
    pairwise_distances = distance(embeddings)

    # Build masks for positive and negative pairs
    positive_mask, negative_mask = build_masks(labels, tf.shape(embeddings)[0])

    # Get positive distances and indices
    positive_dists, positive_indices = positive_distances(
        positive_mining_strategy, pairwise_distances, positive_mask
    )

    # Get negative distances
    negative_dists, _ = negative_distances(
        negative_mining_strategy, pairwise_distances, negative_mask
    )

    # Reorder pairwise distances and negative mask based on positive indices
    reordered_pairwise_distances = tf.gather(pairwise_distances, positive_indices, axis=1)
    reordered_negative_mask = tf.gather(negative_mask, positive_indices, axis=1)

    # Compute (margin - neg_dist) logsum_exp values for each row (equation 4 in the paper)
    neg_logsumexp = tfsim_losses.utils.logsumexp(margin - reordered_pairwise_distances, reordered_negative_mask)

    # Calculate the loss
    j_values = neg_logsumexp + positive_dists
    
    loss = j_values / 2.0

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
        negative_mining_strategy: str = "easy",
        margin: float = 1.0,
        name: str = "LiftedStructLoss",
        **kwargs,
    ):
        """Initializes the LiftedStructLoss.

        Args:
            distance: Which distance function to use to compute the pairwise
                distances between embeddings.
            positive_mining_strategy: What mining strategy to use to select
                embedding from the same class. Defaults to 'hard'.
                Available: {'easy', 'hard'}
            negative_mining_strategy: What mining strategy to use for select the
                embedding from the different class. Defaults to 'easy'.
                Available: {'hard', 'semi-hard', 'easy'}
            margin: Use an explicit value for the margin term.
            name: Loss name. Defaults to "LiftedStructLoss".

        Raises:
            ValueError: Invalid positive mining strategy.
            ValueError: Invalid negative mining strategy.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        # sanity checks
        if positive_mining_strategy not in ["easy", "hard"]:
            raise ValueError("Invalid positive mining strategy")

        if negative_mining_strategy not in ["easy", "hard", "semi-hard"]:
            raise ValueError("Invalid negative mining strategy")

        super().__init__(
            lifted_struct_loss,
            name=name,
            distance=distance,
            positive_mining_strategy=positive_mining_strategy,
            negative_mining_strategy=negative_mining_strategy,
            margin=margin,
            **kwargs,
        )
