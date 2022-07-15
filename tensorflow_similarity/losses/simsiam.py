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
"""SimSiam Loss.
    Exploring Simple Siamese Representation Learning
    https://bit.ly/3LxsWdj
"""
import math
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor


def negative_cosine_sim(sim: FloatTensor) -> FloatTensor:
    loss: FloatTensor = tf.constant([-1.0]) * sim
    return loss


def cosine_distance(sim: FloatTensor) -> FloatTensor:
    loss: FloatTensor = tf.constant([1.0]) - sim
    return loss


def angular_distance(sim: FloatTensor) -> FloatTensor:
    loss: FloatTensor = tf.math.acos(sim) / tf.constant(math.pi)
    return loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SimSiamLoss(tf.keras.losses.Loss):
    """SimSiam Loss.

    Introduced in: [Exploring Simple Siamese Representation Learning](https://bit.ly/3LxsWdj)
    """

    def __init__(
        self,
        projection_type: str = "negative_cosine_sim",
        margin: float = 0.001,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Create the SimSiam Loss.

        Args:
          projection_type: Projects results into a metric space to allow KNN
          search.
            negative_cosine_sim: -1.0 * cosine similarity.
            cosine_distance: 1.0 - cosine similarity.
            angular_distance: 1.0 - angular similarity.
          margin: Offset to prevent a distance of 0.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.projection_type = projection_type
        self.margin = margin

        if self.projection_type == "negative_cosine_sim":
            self._projection = negative_cosine_sim
        elif self.projection_type == "cosine_distance":
            self._projection = cosine_distance
        elif self.projection_type == "angular_distance":
            self._projection = angular_distance
        else:
            raise ValueError(f"{self.projection_type} is not supported.")

    def call(self, projector: FloatTensor, predictor: FloatTensor) -> FloatTensor:
        """Compute the loss.

        Notes:
        - Stopping the gradient is critical according to the paper for convergence.

        Args:
            projector: Projector outputs
            predictor: Predictor outputs

        Returns:
            The per example distance between projector_i and predictor_i.
        """

        projector = tf.math.l2_normalize(projector, axis=1)
        predictor = tf.math.l2_normalize(predictor, axis=1)

        # 2D tensor
        vals = predictor * projector
        # 1D tensor
        cosine_simlarity = tf.reduce_sum(vals, axis=1)

        per_example_projection = self._projection(cosine_simlarity)

        # 1D tensor
        loss: FloatTensor = tf.math.multiply(per_example_projection, 0.5) + self.margin

        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "projection_type": self.projection_type,
            "margin": self.margin,
        }
        base_config = super().get_config()
        return {**base_config, **config}
