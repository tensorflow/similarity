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

"Metrics computed over embeddings"
import tensorflow as tf
from tensorflow.keras.metrics import Metric

from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.algebra import masked_min, masked_max
from tensorflow_similarity.distances import distance_canonicalizer


@tf.keras.utils.register_keras_serializable(package="Similarity")
class DistanceMetric(Metric):
    def __init__(self,
                 distance,
                 aggregate='mean',
                 anchor="positive",
                 name=None,
                 positive_mining_strategy="hard",
                 negative_mining_strategy="hard",
                 **kwargs):

        if not name:
            name = "%s_%s" % (aggregate, anchor[:3])
        super().__init__(name=name, **kwargs)

        self.distance = distance_canonicalizer(distance)

        if anchor not in ['positive', 'negative']:
            raise ValueError('Invalid anchor_type')
        self.anchor = anchor

        if positive_mining_strategy not in ['hard', 'semi-hard', 'easy']:
            raise ValueError('Invalid positive_mining_strategy')
        self.positive_mining_strategy = positive_mining_strategy

        if negative_mining_strategy not in ['hard', 'easy']:
            raise ValueError('Invalid positive_mining_strategy')
        self.negative_mining_strategy = negative_mining_strategy

        if aggregate not in ['mean', 'max', 'avg', 'min', 'sum']:
            raise ValueError('Invalid reduction')
        self.aggregate = aggregate

        # result variable
        self.aggregated_distances = tf.Variable(0, dtype='float32')

    def update_state(self, labels, embeddings, sample_weight):

        # [distances]
        pairwise_distances = self.distance(embeddings)

        # [mask]
        batch_size = tf.size(labels)
        positive_mask, negative_mask = build_masks(labels, batch_size)

        if self.anchor == "positive":
            if self.positive_mining_strategy == "hard":
                distances, _ = masked_max(pairwise_distances, positive_mask)
            else:
                distances, _ = masked_min(pairwise_distances, positive_mask)
        else:
            if self.negative_mining_strategy == 'hard':
                distances, _ = masked_min(pairwise_distances, negative_mask)
            else:
                distances, _ = masked_max(pairwise_distances, negative_mask)

        # reduce
        if self.aggregate == 'mean' or self.aggregate == 'avg':
            aggregated_distances = tf.reduce_mean(distances)
        elif self.aggregate == 'max':
            aggregated_distances = tf.reduce_max(distances)
        elif self.aggregate == 'min':
            aggregated_distances = tf.reduce_min(distances)
        elif self.aggregate == 'sum':
            aggregated_distances = tf.reduce_sum(distances)

        self.aggregated_distances = aggregated_distances

    def reset_state(self):
        self.aggregated_distances = tf.Variable(0, dtype='float32')

    def result(self):
        return self.aggregated_distances

    def get_config(self):
        config =  {
            "distance": self.distance.name,
            "aggregate": self.aggregate,
            "anchor": self.anchor,
            "positive_mining_strategy": self.positive_mining_strategy,
            "negative_mining_strategy": self.negative_mining_strategy
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Similarity")
class DistanceGapMetric(Metric):
    def __init__(self,
                 distance,
                 name=None,
                 **kwargs):
        name = name if name else 'dist_gap'
        super().__init__(name=name, **kwargs)
        self.distance = distance
        self.max_pos_fn = DistanceMetric(distance, aggregate='max')
        self.min_neg_fn = DistanceMetric(distance,
                                         aggregate='min',
                                         anchor='negative')
        self.gap = tf.Variable(0, dtype='float32')

    def update_state(self, labels, embeddings, sample_weight):
        max_pos = self.max_pos_fn(labels, embeddings, sample_weight)
        min_neg = self.min_neg_fn(labels, embeddings, sample_weight)
        self.gap.assign(tf.abs(min_neg - max_pos))

    def result(self):
        return self.gap

    def get_config(self):
        config = {
            "distance": self.distance,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# aliases
@tf.keras.utils.register_keras_serializable(package="Similarity")
def dist_gap(distance):
    return DistanceGapMetric(distance)


# max
@tf.keras.utils.register_keras_serializable(package="Similarity")
def max_pos(distance):
    return DistanceMetric(distance, aggregate='max')


@tf.keras.utils.register_keras_serializable(package="Similarity")
def max_neg(distance):
    return DistanceMetric(distance, aggregate='max', anchor='negative')


# avg
@tf.keras.utils.register_keras_serializable(package="Similarity")
def avg_pos(distance):
    return DistanceMetric(distance, aggregate='mean')


@tf.keras.utils.register_keras_serializable(package="Similarity")
def avg_neg(distance):
    return DistanceMetric(distance, aggregate='mean', anchor='negative')


# min
@tf.keras.utils.register_keras_serializable(package="Similarity")
def min_pos(distance):
    return DistanceMetric(distance, aggregate='min')


@tf.keras.utils.register_keras_serializable(package="Similarity")
def min_neg(distance):
    return DistanceMetric(distance, aggregate='min', anchor='negative')


# sum
@tf.keras.utils.register_keras_serializable(package="Similarity")
def sum_pos(distance):
    return DistanceMetric(distance, aggregate='sum')


@tf.keras.utils.register_keras_serializable(package="Similarity")
def sum_neg(distance):
    return DistanceMetric(distance, aggregate='sum', anchor='negative')
