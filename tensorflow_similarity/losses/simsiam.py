import math
from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.keras.losses import Loss

from tensorflow_similarity.types import FloatTensor


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
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.loss_type = loss_type

    def call(self, z: FloatTensor, p: FloatTensor) -> FloatTensor:
        """Compute the loss
        Notes:
        - Stopping the gradient is critical according to the paper for convergence.

        Args:
            z: Encoder outputs
            p: Predictor outputs
            loss_type: Determines the way the final loss is computed between views.
                negative_cosine_sim: -1.0 * cosine similarity.
                cosine_distance: 1.0 - cosine similarity.
                angular_distance: 1.0 - angular similarity.

        Returns:
            The per example distance between z_i and p_i.
        """
        z = tf.stop_gradient(z)

        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)

        vals = p * z
        sim = tf.reduce_sum(vals, axis=1)

        if self.loss_type == "negative_cosine_sim":
            loss = -1.0 * sim
        elif self.loss_type == "cosine_distance":
            loss = 1.0 - sim
        elif self.loss_type == "angular_distance":
            loss = tf.math.acos(sim) / tf.constant(math.pi)
        else:
            raise ValueError(f"{self.loss_type} is not supported.")

        return loss * 0.5

    def to_config(self) -> Dict[str, Any]:
        return {
            "loss_type": self.loss_type,
        }
