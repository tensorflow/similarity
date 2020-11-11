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
from .metrics import metric_name_canonializer, pairwise_cosine


def _masked_maximum(distances, mask, dim=1):
    """Computes the maximum values over masked pairwise distances.

    We need to use this formula to make sure all values are >=0.

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
      Tensor: The minimal distance value
    """
    axis_max = tf.math.reduce_max(data, dim, keepdims=True)
    masked_min = tf.math.multiply(data - axis_max, mask)
    masked_min = tf.math.reduce_min(masked_min, dim, keepdims=True)
    masked_min = masked_min + axis_max
    return masked_min


def _build_masks(labels, batch_size):
    # same class mask
    positive_mask = tf.math.equal(labels, tf.transpose(labels))
    # not the same class
    negative_mask = tf.math.logical_not(positive_mask)

    # masks are treated as float32 moving forward
    positive_mask = tf.cast(positive_mask, tf.float32)
    negative_mask = tf.cast(negative_mask, tf.float32)

    # we need to remove the diagonal from postivie mask
    diag = tf.linalg.diag(tf.ones(batch_size))
    positive_mask = positive_mask - tf.cast(diag, tf.float32)
    return positive_mask, negative_mask

@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def triplet_loss(labels, embeddings, distance='cosine',
                 positive_mining_strategy='hard',
                 negative_mining_strategy='semi-hard',
                 soft_margin=False,
                 margin=1.0):
    """Triplet loss computations

    Args:
        labels (list(int)): labels associted with the embed
        embeddings ([type]): [description]
        distance (str, optional): [description]. Defaults to 'cosine'.
        positive_mining_strategy (str, optional): [description]. Defaults to 'hard'.
        negative_mining_strategy (str, optional): [description]. Defaults to 'semi-hard'.
        soft_margin (bool, optional): [description]. Defaults to False.
        margin (float, optional): [description]. Defaults to 1.0.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    # [Label]
    # ! do not remove this code. It is actually needed for specific situation
    # Reshape label tensor to [batch_size, 1] if not already in that format.
    labels = tf.reshape(labels, (labels.shape[0], 1))
    batch_size = tf.size(labels)

    # [distances]
    if distance == 'cosine':
        pairwise_distances = pairwise_cosine(embeddings)
    else:
        # user supplied distance function
        pairwise_distances = distance(embeddings)

    # [masks]
    positive_mask, negative_mask = _build_masks(labels, batch_size)

    # [Positivie distance computation]
    if positive_mining_strategy == 'hard':
        positive_distances = _masked_maximum(pairwise_distances, positive_mask)
    elif positive_mining_strategy == 'easy':
        positive_distances = _masked_minimum(pairwise_distances, positive_mask)
    else:
        raise ValueError('Invalid positive mining strategy')

    # [Negative distances computation]
    if negative_mining_strategy == 'hard':
        # find the *non-zero* minimal distance between negative labels
        negative_distances = _masked_minimum(pairwise_distances, negative_mask)
    elif negative_mining_strategy == 'semi-hard':
        # find the minimal distance between negative label gt than max distance
        # between positive labels
        # find max value of positive distance
        max_positive = _masked_maximum(pairwise_distances, positive_mask)

        # select distance that are above the max positive distance
        greater_distances = tf.math.greater(pairwise_distances, max_positive)

        # combine with negative mask: keep negative value if greater,
        # zero otherwise
        empty = tf.cast(tf.zeros((batch_size, batch_size)), tf.dtypes.float32)
        semi_hard_mask = tf.where(greater_distances, negative_mask, empty)

        # find the  minimal distance between negative labels above threshold
        negative_distances = _masked_minimum(pairwise_distances,
                                             semi_hard_mask)

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
        triplet_loss = tf.math.subtract(positive_distances, negative_distances)
        triplet_loss = tf.math.add(triplet_loss, margin)
        triplet_loss = tf.maximum(triplet_loss, 0.0)  # numeric stability

    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss


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
                 soft_margin=False,
                 margin=1.0,
                 name=None):
        """Initializes the TripletLoss

        Args:
            distance (str, optional): Which distance function to use to compute
            the pairwise distances between embeddings. Defaults to 'cosine'.

            positive_mining_strategy (str, optional): What mining strategy to
            use to select embedding from the same class. Defaults to 'hard'.
            available: {'easy', 'hard'}

            negative_mining_strategy (str, optional): What mining strategy to
            use for select the embedding from the differents class.
            Defaults to 'semi-hard'. Available: {'hard', 'semi-hard', 'easy'}

            soft_margin (bool, optional): [description]. Defaults to True.
            Use a soft margin instad of an explict one.

            margin (float, optional): Use an explicit value for the margin
            term. Defaults to 1.0.

            reducer (str, optional): How to accumulate the triplet values
            as a single loss value. Defaults to 'sum'.

            name (str, optional): Loss name. Defaults to None.
        """

        # distance canonicalization
        distance = metric_name_canonializer(distance)

        # sanity checks

        if positive_mining_strategy not in ['easy', 'hard']:
            raise ValueError('Invalid positive mining strategy')

        if negative_mining_strategy not in ['easy', 'hard', 'semi-hard']:
            raise ValueError('Invalid negative mining strategy')

        # Ensure users knows its one or the other
        if margin != 1.0 and soft_margin:
            raise ValueError('Margin value is not used when soft_margin is\
                set to True')

        super().__init__(triplet_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         positive_mining_strategy=positive_mining_strategy,
                         negative_mining_strategy=negative_mining_strategy,
                         soft_margin=soft_margin,
                         margin=margin)
