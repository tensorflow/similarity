import tensorflow as tf
from tensorflow import Tensor
from typing import List, Any, Dict
from tensorflow_similarity.types import FloatTensor
from tensorflow.keras.losses import Loss

# FIXME: make sure to register
#@tf.keras.utils.register_keras_serializable(package="Similarity")
class SimCLRLoss(Loss):
    """SimCLR Loss
    # FIXME original reference
    used in [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)
    code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)
    """
    def __init__(self,
                 temperature: float = 1.0,
                 use_hidden_norm: bool = True,
                 **kwargs):
        super.__init__(kwargs)
        self.temperature = tf.constant(temperature, dtype='float32')
        self.use_hidden_norm = use_hidden_norm

    def call(self,
             zs: List[FloatTensor],
             ps: List[FloatTensor]) -> FloatTensor:
        """compute the loast.
        Args:
        z: Encoders outputs
        p: Projectors outputs

        Returns:
        loss
        """
        # unpack
        za, zb = zs

        batch_size = tf.size(za)

        # compute pairwise
        pairwise = tf.matmul(za, zb, transpose_b=True)

        # divide by temperature once to go faster
        pairwise = pairwise / self.temperature

        # compute the diagonal
        diag = tf.linalg.diag(tf.ones(batch_size, dtype=tf.float32))
        not_diag = 1.0 - diag

        # numerator / denominator
        numerator = tf.exp(tf.reduce_sum(pairwise * diag, axis=1) + 1e-11)

        denominator = tf.exp(pairwise * not_diag + 1e-11)
        denominator = tf.reduce_sum(denominator, axis=1) - 1  # remove the diag

        return 1

    def to_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "use_hidden_norm": self.use_hidden_norm
        }





# def SimCLRLoss(z: List[Tensor], p: List[Tensor]) -> FloatTensor:
#     """SimCLR Loss

#     Introduced in
#     [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)

#     code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)

#     Args:
#         z: Encoders outputs
#         p: Projectors outputs

#     Returns:
#         loss
#     """


#     p = tf.math.l2_normalize(p, axis=1)
#     z = tf.math.l2_normalize(z, axis=1)

#     vals = -p * z
#     vals = tf.reduce_sum(vals, axis=1)
#     vals = tf.reduce_mean(vals, axis=0)
#     return vals + 1 # adding 1 to go from [0, 1] not [-1, 0]