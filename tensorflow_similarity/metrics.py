import tensorflow as tf


def pairwise_cosine(embeddings, axis=1):
    tensor = tf.nn.l2_normalize(embeddings, axis=axis)
    distance = 1 - tf.matmul(tensor, tensor, transpose_b=True)
    return distance


def pairwise_snr():
    """ Signal to Noise pairwise distance

    Proposed in:
    Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    """
    pass
