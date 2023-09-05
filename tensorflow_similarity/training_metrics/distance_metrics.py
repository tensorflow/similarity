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
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tensorflow as tf
from tensorflow.keras.metrics import Metric

import tensorflow_similarity.distances
from tensorflow_similarity.algebra import build_masks, masked_max, masked_min

if TYPE_CHECKING:
    from tensorflow_similarity.distances import Distance
    from tensorflow_similarity.types import FloatTensor, IntTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class DistanceMetric(Metric):
    def __init__(
        self,
        distance: Distance | str,
        aggregate: str = "mean",
        anchor: str = "positive",
        name: str | None = None,
        positive_mining_strategy: str = "hard",
        negative_mining_strategy: str = "hard",
        **kwargs,
    ):
        if not name:
            name = "%s_%s" % (aggregate, anchor[:3])
        super().__init__(name=name, **kwargs)

        self.distance = tensorflow_similarity.distances.get(distance)

        if anchor not in ["positive", "negative"]:
            raise ValueError("Invalid anchor_type")
        self.anchor = anchor

        if positive_mining_strategy not in ["hard", "semi-hard", "easy"]:
            raise ValueError("Invalid positive_mining_strategy")
        self.positive_mining_strategy = positive_mining_strategy

        if negative_mining_strategy not in ["hard", "easy"]:
            raise ValueError("Invalid positive_mining_strategy")
        self.negative_mining_strategy = negative_mining_strategy

        if aggregate not in ["mean", "max", "avg", "min", "sum"]:
            raise ValueError("Invalid reduction")
        self.aggregate = aggregate

        # result variable
        self.aggregated_distances = tf.Variable(0, dtype=tf.keras.backend.floatx())

    def update_state(self, labels: IntTensor, embeddings: FloatTensor, sample_weight: FloatTensor) -> None:
        # [distances]
        pairwise_distances = self.distance(embeddings, embeddings)

        # [mask]
        batch_size = tf.size(labels)
        positive_mask, negative_mask = build_masks(labels, labels, batch_size)

        if self.anchor == "positive":
            if self.positive_mining_strategy == "hard":
                distances, _ = masked_max(pairwise_distances, positive_mask)
            else:
                distances, _ = masked_min(pairwise_distances, positive_mask)
        else:
            if self.negative_mining_strategy == "hard":
                distances, _ = masked_min(pairwise_distances, negative_mask)
            else:
                distances, _ = masked_max(pairwise_distances, negative_mask)

        # reduce
        if self.aggregate == "mean" or self.aggregate == "avg":
            aggregated_distances = tf.reduce_mean(distances)
        elif self.aggregate == "max":
            aggregated_distances = tf.reduce_max(distances)
        elif self.aggregate == "min":
            aggregated_distances = tf.reduce_min(distances)
        elif self.aggregate == "sum":
            aggregated_distances = tf.reduce_sum(distances)

        self.aggregated_distances = aggregated_distances

    def reset_state(self) -> None:
        self.aggregated_distances = tf.Variable(0, dtype=tf.keras.backend.floatx())

    def result(self) -> tf.Variable:
        return self.aggregated_distances

    def get_config(self) -> dict[str, Any]:
        config = {
            "distance": self.distance.name,
            "aggregate": self.aggregate,
            "anchor": self.anchor,
            "positive_mining_strategy": self.positive_mining_strategy,
            "negative_mining_strategy": self.negative_mining_strategy,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Similarity")
class DistanceGapMetric(Metric):
    def __init__(self, distance: Distance | str, name: str | None = None, **kwargs):
        name = name if name else "dist_gap"
        super().__init__(name=name, **kwargs)
        self.distance = distance
        self.max_pos_fn = DistanceMetric(distance, aggregate="max")
        self.min_neg_fn = DistanceMetric(distance, aggregate="min", anchor="negative")
        self.gap = tf.Variable(0, dtype=tf.keras.backend.floatx())

    def update_state(self, labels: IntTensor, embeddings: FloatTensor, sample_weight: FloatTensor):
        max_pos = self.max_pos_fn(labels, embeddings, sample_weight)
        min_neg = self.min_neg_fn(labels, embeddings, sample_weight)
        self.gap.assign(tf.cast(tf.abs(min_neg - max_pos), tf.keras.backend.floatx()))

    def result(self) -> tf.Variable:
        return self.gap

    def get_config(self) -> dict[str, Any]:
        config = {
            "distance": self.distance,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# aliases
@tf.keras.utils.register_keras_serializable(package="Similarity")
def dist_gap(distance: Distance | str) -> DistanceGapMetric:
    return DistanceGapMetric(distance)


# max
@tf.keras.utils.register_keras_serializable(package="Similarity")
def max_pos(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="max")


@tf.keras.utils.register_keras_serializable(package="Similarity")
def max_neg(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="max", anchor="negative")


# avg
@tf.keras.utils.register_keras_serializable(package="Similarity")
def avg_pos(distance: Distance | str) -> DistanceGapMetric:
    return DistanceMetric(distance, aggregate="mean")


@tf.keras.utils.register_keras_serializable(package="Similarity")
def avg_neg(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="mean", anchor="negative")


# min
@tf.keras.utils.register_keras_serializable(package="Similarity")
def min_pos(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="min")


@tf.keras.utils.register_keras_serializable(package="Similarity")
def min_neg(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="min", anchor="negative")


# sum
@tf.keras.utils.register_keras_serializable(package="Similarity")
def sum_pos(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="sum")


@tf.keras.utils.register_keras_serializable(package="Similarity")
def sum_neg(distance: Distance | str) -> DistanceMetric:
    return DistanceMetric(distance, aggregate="sum", anchor="negative")
