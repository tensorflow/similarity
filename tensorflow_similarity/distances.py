import tensorflow as tf
from .types import FloatTensorLike


def metric_name_canonializer(metric_name: str) -> str:
    """Normalize metric name as each have various names in the literature

    Args:
        metric_name: name of the metric to canonicalize.
    """
    metric_name = metric_name.lower().strip()

    mapping = {
        "l2": 'cosine',
        'cosine': 'cosine'
    }

    if metric_name not in mapping:
        raise ValueError('Metric not supported by the framework')

    return mapping[metric_name]


@tf.function
def pairwise_euclidean(embeddings: FloatTensorLike,
                       axis: int = 1) -> FloatTensorLike:
    squared_norm = tf.math.square(embeddings)
    squared_norm = tf.math.reduce_sum(squared_norm, axis=axis, keepdims=True)

    distances = 2.0 * tf.linalg.matmul(embeddings,
                                       embeddings,
                                       transpose_b=True)
    distances = squared_norm - distances + tf.transpose(squared_norm)

    # Avoid NaN gradients when back propegating through the sqrt.
    distances = tf.math.maximum(distances, 1e-16)
    distances = tf.math.sqrt(distances)

    return distances


@tf.function
def pairwise_cosine(embeddings: FloatTensorLike,
                    axis: int = 1) -> FloatTensorLike:
    tensor = tf.nn.l2_normalize(embeddings, axis=axis)
    distances = 1 - tf.linalg.matmul(tensor, tensor, transpose_b=True)
    distances = tf.math.maximum(distances, 0.0)
    return distances


@tf.function
def cosine(a: FloatTensorLike,
           b: FloatTensorLike,
           axis: int = -1) -> FloatTensorLike:
    t1 = tf.nn.l2_normalize(a, axis=axis)
    t2 = tf.nn.l2_normalize(b, axis=axis)
    distances = 1 - tf.linalg.matmul(t1, t2, transpose_b=True)
    distances = tf.math.maximum(distances, 0.0)
    return distances


def pairwise_snr():
    """ Signal to Noise pairwise distance

    Proposed in:
    Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    """
    pass
