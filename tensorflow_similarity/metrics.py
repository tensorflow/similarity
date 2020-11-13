import tensorflow as tf


def metric_name_canonializer(metric_name):
    """Normalize metric name as each have various names in the litterature

    Args:
        metric_name (str): name of the metric to canonialize.
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
def pairwise_cosine(embeddings, axis=1):
    tensor = tf.nn.l2_normalize(embeddings, axis=axis)
    distances = 1 - tf.matmul(tensor, tensor, transpose_b=True)
    distances = tf.maximum(distances, 0.0)
    return distances


@tf.function
def cosine(a, b, axis=-1):
    t1 = tf.nn.l2_normalize(a, axis=axis)
    t2 = tf.nn.l2_normalize(b, axis=axis)
    distances = 1 - tf.matmul(t1, t2, transpose_b=True)
    distances = tf.maximum(distances, 0.0)
    return distances


def pairwise_snr():
    """ Signal to Noise pairwise distance

    Proposed in:
    Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    """
    pass
