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
import tensorflow as tf
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from .metrics import pairwise_cosine


def _masked_maximum(distances, mask, dim=1):
    """Computes the maximum values over masked pairwise distances.

    Args:
      distances (Tensor): 2-D float `Tensor` of [n, n] pairwise distances
      mask (Tensor): 2-D Boolean `Tensor` of [n, n] valid distance size.
      dim: The dimension over which to compute the maximum.

    Returns:
      Tensor: The maximum distance value
    """
    axis_min = tf.math.reduce_min(distances, dim, keepdims=True)
    masked_max = tf.math.multiply(distances - axis_min, mask)
    masked_max = tf.math.reduce_max(masked_max, dim, keepdims=True) + axis_min
    return masked_max


def _masked_minimum(data, mask, dim=1):
    """Computes the mimimal values over masked pairwise distances.

    Args:
      distances (Tensor): 2-D float `Tensor` of [n, n] pairwise distances
      mask (Tensor): 2-D Boolean `Tensor` of [n, n] valid distance size.
      dim: The dimension over which to compute the maximum.

    Returns:
      Tensor: The maximum distance value
    """
    axis_max = tf.math.reduce_max(data, dim, keepdims=True)
    masked_min = tf.math.multiply(data - axis_max, mask)
    masked_min = tf.math.reduce_min(masked_min, dim, keepdims=True) + axis_max
    return masked_min


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def triplet_loss(labels, embeddings, distance='cosine',
                 positive_mining_strategy='hard',
                 negative_mining_strategy='hard',
                 soft_margin=True,
                 margin=0.5,
                 reducer='sum'):

    # Ensure users knows its one or the other
    if margin != 0.5 and soft_margin:
        raise ValueError('Margin value is not used when soft_margin is True')

    # [Label]
    labels = tf.expand_dims(labels, axis=1)

    # [masks]
    positive_mask = tf.math.equal(labels, tf.transpose(labels))
    negative_mask = tf.math.logical_not(positive_mask)

    # masks are treated as float32 moving forward
    positive_mask = tf.cast(positive_mask, tf.float32)
    negative_mask = tf.cast(negative_mask, tf.float32)

    # we need to remove the diagonal from postivie mask
    diag = tf.linalg.diag(tf.ones(labels.shape[0]))
    positive_mask = positive_mask - diag

    # [distances]
    # comptutet pairwise distance > [batch_size, batch_size]
    if distance == 'cosine':
        pairwise_distances = pairwise_cosine(embeddings)
    else:
        raise ValueError('Invalid distance')

    # Ensure numeric stability
    pairwise_distances = tf.maximum(pairwise_distances, 0.0)

    # [Positive distances computation]

    # select positive distance based of mining strategy
    if positive_mining_strategy == 'hard':
        positive_distances = _masked_maximum(pairwise_distances, positive_mask)
    elif positive_mining_strategy == 'easy':
        positive_distances = _masked_minimum(pairwise_distances, positive_mask)
    else:
        raise ValueError('Invalid positive mining strategy')

    # [Negative distances computation]

    # select distance based of mining strategy
    if negative_mining_strategy == 'hard':
        # find the *non-zero* minimal distance between negative labels
        negative_distances = _masked_minimum(pairwise_distances, negative_mask)
    # elif negative_mining_strategy == 'semi-hard':
    # FIXME we need to select min of distance above a threshold
    elif negative_mining_strategy == 'easy':
        # find the maximal distance between negative labels
        negative_distances = _masked_maximum(pairwise_distances, negative_mask)
    else:
        raise ValueError('Invalid negative mining strategy')

    # [Triplet loss computation]
    if soft_margin:
        triplet_loss = tf.math.exp(positive_distances - negative_distances)
        triplet_loss = tf.math.log1p(triplet_loss)
    else:
        triplet_loss = positive_distances - negative_distances + margin
        triplet_loss = tf.maximum(triplet_loss, 0.0)  # numeric stability

    # The original paper use sum as reducer but many implementation use mean
    if reducer == 'sum':
        return tf.reduce_sum(triplet_loss)
    else:
        return tf.reduce_mean(triplet_loss)


@tf.keras.utils.register_keras_serializable(package="Similarity")
class TripletLoss(LossFunctionWrapper):
    """Computes the triplet loss in an online fashion.

    This loss encourages the positive distances between a pair of embeddings
    with the same labels to be smaller than the minimum negative distances
    between pair of embeddings of different labels.
    See: https://arxiv.org/abs/1503.03832 for the original paper.


    `y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """

    def __init__(self,
                 distance='cosine',
                 positive_mining_strategy='hard',
                 negative_mining_strategy='hard',
                 soft_margin=True,
                 margin=0.5,
                 reducer='sum',
                 name=None):
        """Initializes the TripletLoss

        Args:
            distance (str, optional): Which distance function to use to compute
            the pairwise distances between embeddings. Defaults to 'cosine'.

            positive_mining_strategy (str, optional): What mining strategy to
            use to select embedding from the same class. Defaults to 'hard'.

            negative_mining_strategy (str, optional): What mining strategy to
            use for select the embedding from the differents class.
            Defaults to 'hard'.

            soft_margin (bool, optional): [description]. Defaults to True.
            Use a soft margin instad of an explict one.

            margin (float, optional): Use an explicit value for the margin
            term. Defaults to 0.5.

            reducer (str, optional): How to accumulate the triplet values
            as a single loss value. Defaults to 'sum'.

            name (str, optional): Loss name. Defaults to None.
        """
        super().__init__(triplet_loss,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         positive_mining_strategy=positive_mining_strategy,
                         negative_mining_strategy=negative_mining_strategy,
                         soft_margin=soft_margin,
                         margin=margin,
                         reducer=reducer,
                         name=name)
