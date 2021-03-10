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
"""Metric losses

References
    - o
    - : FaceNet: A Unified Embedding for Face Recognition
    and Clustering: https://arxiv.org/abs/1503.03832

    - Mining strategies:
    https://openaccess.thecvf.com/content_WACV_2020/papers/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.pdf

    - https://arxiv.org/pdf/1712.07682.pdf
"""

import tensorflow as tf
from .utils import is_tensor_or_variable
from .distances import Distance, distance_canonicalizer
from .algebra import masked_maximum, masked_minimum, build_masks
from .types import FloatTensor
from typing import Callable, List, Union


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def triplet_loss(labels: List[int],
                 embeddings: FloatTensor,
                 distance: Callable,
                 positive_mining_strategy: str = 'hard',
                 negative_mining_strategy: str = 'semi-hard',
                 soft_margin: bool = False,
                 margin: float = 1.0):
    """Triplet loss computations

    Args:
        labels (list(int)): labels associted with the embed
        embeddings ([type]): [description]
        distance (str, optional): [description]. Defaults to 'cosine'.
        positive_mining_strategy (str, optional): [description].
        Defaults to 'hard'.
        negative_mining_strategy (str, optional): [description].
        Defaults to 'semi-hard'.
        soft_margin (bool, optional): [description]. Defaults to False.
        margin (float, optional): [description]. Defaults to 1.0.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    # [Label]
    # ! Weirdness to be investigated
    # do not remove this code. It is actually needed for specific situation
    # Reshape label tensor to [batch_size, 1] if not already in that format.
    # labels = tf.reshape(labels, (labels.shape[0], 1))
    batch_size = tf.size(labels)

    # [distances]
    pairwise_distances = distance(embeddings)

    # [masks]
    positive_mask, negative_mask = build_masks(labels, batch_size)

    # [Positivie distance computation]
    if positive_mining_strategy == 'hard':
        positive_distances = masked_maximum(pairwise_distances, positive_mask)
    elif positive_mining_strategy == 'easy':
        positive_distances = masked_minimum(pairwise_distances, positive_mask)
    else:
        raise ValueError('Invalid positive mining strategy')

    # [Negative distances computation]
    if negative_mining_strategy == 'hard':
        # find the *non-zero* minimal distance between negative labels
        negative_distances = masked_minimum(pairwise_distances, negative_mask)
    elif negative_mining_strategy == 'semi-hard':
        # find the minimal distance between negative label gt than max distance
        # between positive labels
        # find max value of positive distance
        max_positive = masked_maximum(pairwise_distances, positive_mask)

        # select distance that are above the max positive distance
        greater_distances = tf.math.greater(pairwise_distances, max_positive)

        # combine with negative mask: keep negative value if greater,
        # zero otherwise
        empty = tf.cast(tf.zeros((batch_size, batch_size)), tf.dtypes.float32)
        semi_hard_mask = tf.where(greater_distances, negative_mask, empty)

        # find the  minimal distance between negative labels above threshold
        negative_distances = masked_minimum(pairwise_distances, semi_hard_mask)

    elif negative_mining_strategy == 'easy':
        # find the maximal distance between negative labels
        negative_distances = masked_maximum(pairwise_distances, negative_mask)
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


class MetricLoss(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self,
        fn,
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
        **kwargs
    ):
        """Initializes `LossFunctionWrapper` class.
        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in iter(self._fn_kwargs.items()):
            if is_tensor_or_variable(v):
                config[k] = tf.keras.backend.eval(v)
            else:
                config[k] = v
        config['name'] = self.name

        # FIXME: seems we can't pass reduction why? its not
        # technically needed for now but some other loss might need it
        # config['reduction'] = self.reduction

        return config


@tf.keras.utils.register_keras_serializable(package="Similarity")
class TripletLoss(MetricLoss):
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
                 distance: Union[Distance, str] = 'cosine',
                 positive_mining_strategy: str = 'hard',
                 negative_mining_strategy: str = 'hard',
                 soft_margin: bool = False,
                 margin: float = 1.0,
                 name: str = None):
        """Initializes the TripletLoss

        Args:
            distance (Un, optional): Which distance function to use to compute
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
        distance = distance_canonicalizer(distance)
        self.distance = distance
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
