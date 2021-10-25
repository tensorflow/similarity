import math
from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.keras.losses import Loss

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
class SimSiamLoss(Loss):
    """SimSiam Loss

    Introduced in: [Exploring Simple Siamese Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)
    """

    def __init__(
        self,
        loss_type: str = "negative_cosine_sim",
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Create the SimSiam Loss.

        Args:
          loss_type: Determines the way the final loss is computed between views.
            negative_cosine_sim: -1.0 * cosine similarity.
            cosine_distance: 1.0 - cosine similarity.
            angular_distance: 1.0 - angular similarity.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.loss_type = loss_type

        if self.loss_type == "negative_cosine_sim":
            self._loss = negative_cosine_sim
        elif self.loss_type == "cosine_distance":
            self._loss = cosine_distance
        elif self.loss_type == "angular_distance":
            self._loss = angular_distance
        else:
            raise ValueError(f"{self.loss_type} is not supported.")

    def call(self, z: FloatTensor, p: FloatTensor) -> FloatTensor:
        """Compute the loss
        Notes:
        - Stopping the gradient is critical according to the paper for convergence.

        Args:
            z: Encoder outputs
            p: Predictor outputs

        Returns:
            The per example distance between z_i and p_i.
        """
        z = tf.stop_gradient(z)

        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        vals = p * z
        sim = tf.reduce_sum(vals, axis=1)

        loss: FloatTensor = self._loss(sim) * tf.constant([0.5])
        return loss

    def to_config(self) -> Dict[str, Any]:
        return {
            "loss_type": self.loss_type,
        }
