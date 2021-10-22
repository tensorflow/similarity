import tensorflow as tf


@tf.function
@tf.keras.utils.register_keras_serializable(package="Similarity")
def encoder_std(zs, ps):
    """Measure the representation embedding standard deviation

    Used to measure if the embeddings are collapsing: if equal
    to zero the model learned a degenerated solution.

    Introduced in: [Exploring Simple Siamese Representation Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)
    """
    z = zs[0]  #only use one of the outt
    z_size = tf.cast(tf.shape(z), dtype="float32")
    # only measure stddev on one z as its shared weights.
    stddev = tf.math.reduce_std(tf.math.l2_normalize(z))
    #FIXME: Owen to fix his complicated super duper estiamte
    # #z_shape = tf.shape(zs[0][0])[0]
    #metric = (1/tf.sqrt(z_size) - stddev)

    return stddev