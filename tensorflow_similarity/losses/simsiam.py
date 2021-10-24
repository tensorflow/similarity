import math

import tensorflow as tf
from tensorflow_similarity.types import FloatTensor


@tf.function
def SimSiamLoss(z: FloatTensor, p: FloatTensor, angular: bool = True) -> FloatTensor:
    """SimSiam Loss

    Introduced in: [Exploring Simple Siamese Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)

    Notes:
    - Stopping the gradient is critical according to the paper for convergence.

    Args:
        z: Encoder outputs
        p: Predictor outputs
        angular: If True, convert the cosine similarity to angular similarity

    Returns:
        The per example distance between z_i and p_i.
    """
    z = tf.stop_gradient(z)

    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)

    vals = p * z
    sim = tf.reduce_sum(vals, axis=1)

    if angular:
        sim = (1 - tf.math.acos(sim)) / tf.constant(math.pi)

    distance: FloatTensor = 1 - sim

    return distance
