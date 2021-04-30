import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Similarity")
class MetricEmbedding(Layer):
    def __init__(self, size):
        """Normalized embedding layer
        Args:
            size (int): size of the embbeding. Usually something like 32, 64 or
            256 floats.
        """
        self.size = size
        self.dense = layers.Dense(size)
        super(MetricEmbedding, self).__init__()
        # FIXME: enforce the shape
        # self.input_spec = rank2

    def call(self, inputs):
        x = self.dense(inputs)
        return tf.math.l2_normalize(x, axis=1)

    def get_config(self):
        return {
            'size': self.size
        }
