from tensorflow_similarity.samplers.samplers import Sampler

import tensorflow as tf


def batch_class_ratio(sampler: Sampler, num_batches: int = 100) -> float:
    """Computes the average number of examples per class within each batch.
    Similarity learning requires at least 2 examples per class in each batch.
    This is needed in order to construct the triplets. This function
    provides the average number of examples per class within each batch and
    can be used to check that a sampler is working correctly.
    The ratio should be >= 2.
    Args:
        sampler: A tf.similarity sampler object.
        num_batches: The number of batches to sample.
    Returns:
        The average number of examples per class.
    """
    ratio = 0
    for batch_count, (_, y) in enumerate(sampler):
        if batch_count < num_batches:
            batch_size = tf.shape(y)[0]
            num_classes = tf.shape(tf.unique(y)[0])[0]
            ratio += tf.math.divide(batch_size, num_classes)
        else:
            break

    return float(ratio/(batch_count+1))
