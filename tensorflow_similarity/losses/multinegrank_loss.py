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
"""Multiple negatives ranking loss
    Useful in situations where there are only positive examples
    All other example in a batch is considered as negative
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tensorflow as tf

if TYPE_CHECKING:
    from ..types import FloatTensor
    from ..distances import Distance

from .. import distances
from .metric_loss import MetricLoss
from .utils import logsumexp


def multineg_ranking_loss(
    query_emb: FloatTensor,
    key_emb: FloatTensor,
    scale: float,
    distance: Distance,
) -> Any:
    """Computes the multiple negatives ranking loss.

    Args:
        query_emb: Embedded query examples.
        key_emb: Embedded key examples.
        scale: Float multiplier for scaling loss value
        distance: Which distance function to use to compute the pairwise.

    Returns:
        loss: loss value for the current batch.
    """

    pairwise_sim = distance(query_emb, key_emb) * scale
    if distance.name != "inner_product":
        raise ValueError("Distance must be inner_product")

    sii = tf.linalg.diag_part(pairwise_sim)
    sij_mask = tf.ones(pairwise_sim.shape)
    sij = logsumexp(pairwise_sim, sij_mask)
    examples_loss = -sii + sij

    return examples_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class MultiNegativesRankLoss(MetricLoss):
    """Multiple negatives ranking loss

    This loss is useful in cases where only positive pairs
    of embeddings are available. For each example in a batch
    it considers all other examples as negative examples and
    computes the loss.
    See https://arxiv.org/abs/1705.00652 for the original paper.
    """

    def __init__(
        self,
        distance: Distance | str = "inner_product",
        scale: float = 20,
        name: str = "multineg_rank_loss",
        **kwargs,
    ):
        """Initializes the MultipleNegativesRankLoss Loss

        Args:
            distance: Which distance function to use.
            scale: Float value for scaling the loss value. Defaults to 20
            name: Optional name for the instance. Defaults to 'multineg_rank_loss'.
        """
        # distance canonicalization
        self.distance = distances.get(distance)
        self.scale = scale

        super().__init__(
            multineg_ranking_loss,
            name=name,
            # The following are passed to the multi_neg_loss function as fn_kwargs
            distance=self.distance,
            **kwargs,
        )

    def call(self, query_emb: FloatTensor, key_emb: FloatTensor) -> FloatTensor:
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          query_emb: Query embeddings
          key_emb: Key embeddings

        Returns:
          Loss values per sample.
        """
        loss: FloatTensor = self.fn(query_emb, key_emb, self.scale, **self._fn_kwargs)
        return loss
