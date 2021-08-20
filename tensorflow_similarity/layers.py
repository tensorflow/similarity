from typing import Dict
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from .types import FloatTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class MetricEmbedding(Layer):
    def __init__(self, size: int):
        """Normalized embedding layer
        Args:
            size: size of the embbeding. Usually something like 32, 64 or
            256 floats.
        """
        self.size = size
        self.dense = layers.Dense(size)
        super().__init__()
        # FIXME: enforce the shape
        # self.input_spec = rank2

    def call(self, inputs: FloatTensor) -> FloatTensor:
        x = self.dense(inputs)
        normed_x: FloatTensor = tf.math.l2_normalize(x, axis=1)
        return normed_x

    def get_config(self) -> Dict[str, int]:
        return {'size': self.size}
