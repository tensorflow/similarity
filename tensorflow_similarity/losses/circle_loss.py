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
"""Circle Loss
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857
"""

import tensorflow as tf
from typing import Callable, Union

from tensorflow.python.framework.ops import get_name_scope

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.types import FloatTensor, IntTensor
from .utils import negative_distances, positive_distances, compute_loss
from .metric_loss import MetricLoss


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def circle_loss(labels: IntTensor,
                embeddings: FloatTensor,
                distance: Callable,
                gamma: float = 128,
                margin: float = 0.25):
    """Circle loss computations

    Args:
        labels: Labels associated with the embeddings

        embeddings: Embeddings as infered by the model.

        distance: Distance function to use to compute the pairwise.

        gamma: Scaling term. Defaults to 256. Good starting value
        according to figure 3. Should be hypertuned.

        margin: Distance minimal margin. Defaults to 0.25

    Returns:
        loss
    """

    # [delta]
    delta_pos = 1 - margin
    delta_neg = margin

    # [labels]
    batch_size = tf.size(labels)

    # [distances]
    pairwise_distances = distance(embeddings)

    # [masks]
    positive_mask, negative_mask = build_masks(labels, batch_size)

    # [flattening]
    flat_shape = [batch_size *  batch_size]

    zeros_tensor = tf.zeros(flat_shape)
    flat_pairwise = tf.reshape(pairwise_distances, flat_shape)
    flat_pos_mask = tf.reshape(positive_mask, flat_shape)
    flat_neg_mask = tf.reshape(negative_mask, flat_shape)

    # [posivites]
    # FIXME: more efficient way to select only positive in 1d dim?
    pos_dists = tf.where(flat_pos_mask, flat_pairwise, zeros_tensor)
    opti_pos = pos_dists * -1 + 1  # inverting distance
    opti_pos = opti_pos + margin  # add margin
    opti_pos = tf.clip_by_value(opti_pos, 0, 1 + margin)  # clip
    opti_pos = tf.where(flat_pos_mask, opti_pos, zeros_tensor)
    opti_pos *= -1  # flip again

    # positive logits
    logit_pos = opti_pos * (pos_dists - delta_pos) * gamma
    logit_pos = tf.reduce_logsumexp(logit_pos)

    # [negatives]
    # FIXME: more efficient way to select only negative in 1d dim?
    neg_dists = tf.where(flat_neg_mask, flat_pairwise, zeros_tensor)
    opti_neg = neg_dists + margin  # add margin
    opti_neg = tf.where(flat_neg_mask, opti_neg, zeros_tensor)

    # negative logits
    logit_neg = opti_neg * (neg_dists - delta_neg) * gamma
    logit_neg = tf.reduce_logsumexp(logit_neg)

    # compute the loss
    loss = tf.nn.softplus(logit_neg + logit_pos)
    return loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class CircleLoss(MetricLoss):
    """Computes the CircleLoss

    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857


    `y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """
    def __init__(self,
                 distance: Union[Distance, str] = 'cosine',
                 gamma: float = 256,
                 margin: float = 0.25,
                 name: str = None):
        """Initializes a CircleLoss

        Args:
            distance: Which distance function to use to compute
            the pairwise distances between embeddings. Defaults to 'cosine'.

            gamma: Scaling term. Defaults to 256. Good starting value
            according to figure 3. Should be hypertuned.

            margin: Margin term. Defaults to 0.25 as in the paper.

            name: Loss name. Defaults to None.

        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(circle_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         gamma=gamma,
                         margin=margin)
