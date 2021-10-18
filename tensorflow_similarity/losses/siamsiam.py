import tensorflow as tf
from tensorflow import Tensor
from typing import List
from tensorflow_similarity.types import FloatTensor


@tf.function
def SiamSiamLoss(z: List[Tensor], p: List[Tensor]) -> FloatTensor:
    """SiamSiam Loss

    Introduced in: [Exploring Simple Siamese Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)

    Notes:
    - Stopping the gradient is critical according to the paper for convergence.
    - This is a modified version that uses cosine similarity and not negative
    cosine similarity to ensure the final loss is between [0,1]

    Args:
        z: Encoders outputs
        p: Projectors outputs

    Returns:
        loss
    """
    z = tf.stop_gradient(z)

    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

    vals = -p * z
    vals = tf.reduce_sum(vals, axis=1)
    vals = tf.reduce_mean(vals, axis=0)
    return vals + 1  # adding 1 to go from [0, 1] not [-1, 0]
