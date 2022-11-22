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
"""PN Loss foudn in Ivis.
    Szubert, B., Cole, J.E., Monaco, C. et al.
    Structure-preserving visualisation of high dimensional single-cell dataset
    Sci Rep 9, 8914 (2019). https://doi.org/10.1038/s41598-019-45301-0
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf

from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor

from .metric_loss import MetricLoss
from .utils import compute_loss, negative_distances, positive_distances


def pn_loss(
    query_labels: IntTensor,
    query_embeddings: FloatTensor,
    key_labels: IntTensor,
    key_embeddings: FloatTensor,
    distance: Distance,
    remove_diagonal: bool = True,
    positive_mining_strategy: str = "hard",
    negative_mining_strategy: str = "semi-hard",
    soft_margin: bool = True,
    margin: float = 0.1,
) -> Any:
    """Positive Negative loss computations.

    Based on the pn loss used in IVIS.

    Args:
        query_labels: labels associated with the query embed.
        query_embeddings: Embedded query examples.
        key_labels: labels associated with the key embed.
        key_embeddings: Embedded key examples.
        distance: The distance function to use to compute the pairwise distances
            between embeddings.
        remove_diagonal: If True, set the diagonal to False in the positive pair
            mask
        positive_mining_strategy: What mining strategy to use to select
            embedding from the same class. {'easy', 'hard'}
        negative_mining_strategy: What mining strategy to use for select the
            embedding from the different class. {'hard', 'semi-hard', 'easy'}
        soft_margin: If True, use a soft margin instead of an explicit one.
        margin: Use an explicit value for the margin term.

    Returns:
        Loss: The loss value for the current batch.
    """

    # [Label]
    # ! Weirdness to be investigated
    # do not remove this code. It is actually needed for specific situation
    # Reshape label tensor to [batch_size, 1] if not already in that format.
    # labels = tf.reshape(labels, (labels.shape[0], 1))
    batch_size = tf.size(query_labels)

    # [distances]
    pairwise_distances = distance(query_embeddings, key_embeddings)

    # [masks]
    positive_mask, negative_mask = build_masks(
        query_labels,
        key_labels,
        batch_size=batch_size,
        remove_diagonal=remove_diagonal,
    )

    # [Positive distance computation]
    pos_distances, pos_idxs = positive_distances(
        positive_mining_strategy,
        pairwise_distances,
        positive_mask,
    )

    # [Negative distances computation]
    neg_distances, neg_idxs = negative_distances(
        negative_mining_strategy,
        pairwise_distances,
        negative_mask,
        positive_mask,
    )

    # Compute the distance between the pairs of positive and negative examples.
    # Then take the min(pos_neg_dist, anchor_neg_dist) as the neg_distances.
    # This encourages both the anchor and the positives to be far from the
    # negative.
    pn_pairs = tf.stack([pos_idxs, neg_idxs], axis=1)
    pn_distances = tf.gather_nd(pairwise_distances, pn_pairs)
    pn_distances = tf.reshape(pn_distances, [-1, 1])
    neg_distances = tf.math.minimum(pn_distances, neg_distances)

    # [PN loss computation]
    pn_loss = compute_loss(pos_distances, neg_distances, soft_margin, margin)

    return pn_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class PNLoss(MetricLoss):
    """Computes the PN loss in an online fashion.

    This loss encourages the positive distances between a pair of embeddings
    with the same labels to be smaller than the minimum negative distances
    between pair of embeddings of different labels. Additionally, both the
    anchor and the positive embeddings are encouraged to be far from the
    negative embeddings. This is accomplished by taking the
    min(pos_neg_dist, anchor_neg_dist) and using that as the negative distance
    in the triplet loss.

    See PN Loss Ivis:
    Szubert, B., Cole, J.E., Monaco, C. et al.
    Structure-preserving visualisation of high dimensional single-cell dataset
    Sci Rep 9, 8914 (2019). https://doi.org/10.1038/s41598-019-45301-0

    `y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """

    def __init__(
        self,
        distance: Distance | str = "cosine",
        positive_mining_strategy: str = "hard",
        negative_mining_strategy: str = "semi-hard",
        soft_margin: bool = True,
        margin: float = 0.1,
        name: str = "PNLoss",
        **kwargs,
    ):
        """Initializes the PN Loss.

        Args:
            distance: The distance function to use to compute the pairwise
                distances between embeddings.
            positive_mining_strategy: What mining strategy to use to select
                embedding from the same class. {'easy', 'hard'}
            negative_mining_strategy: What mining strategy to use for select the
                embedding from the different class. {'hard', 'semi-hard', 'easy'}
            soft_margin: If True, use a soft margin instead of an explicit one.
            margin: Use an explicit value for the margin term.
            name: Loss name.

        Raises:
            ValueError: Invalid positive mining strategy.
            ValueError: Invalid negative mining strategy.
            ValueError: Margin value is not used when soft_margin is set to True.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance
        # sanity checks

        if positive_mining_strategy not in ["easy", "hard"]:
            raise ValueError("Invalid positive mining strategy.")

        if negative_mining_strategy not in ["easy", "hard", "semi-hard"]:
            raise ValueError("Invalid negative mining strategy.")

        # Ensure users knows its one or the other
        if margin != 0.1 and soft_margin:
            raise ValueError("Margin value is not used when soft_margin is set to True")

        super().__init__(
            pn_loss,
            name=name,
            distance=distance,
            positive_mining_strategy=positive_mining_strategy,
            negative_mining_strategy=negative_mining_strategy,
            soft_margin=soft_margin,
            margin=margin,
            **kwargs,
        )
