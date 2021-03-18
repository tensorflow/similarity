from abc import abstractmethod, ABC
from typing import Union
import tensorflow as tf
from .types import FloatTensor


class Distance(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        """Compute distance"""

    def __call__(self, embeddings: FloatTensor):
        return self.call(embeddings)

    def __str__(self) -> str:
        return self.name

    def get_config(self):
        return {
            "name": self.name
        }

@tf.keras.utils.register_keras_serializable(package="Similarity")
class CosineDistance(Distance):

    def __init__(self, name: str = None):
        """Compute pairwise cosine distances"""
        name = name if name else 'cosine'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        print(embeddings.shape)
        x_rs = tf.reshape(embeddings, shape=[tf.shape(embeddings)[0], -1])
        tensor = tf.nn.l2_normalize(x_rs, axis=1)
        distances: FloatTensor = 1 - tf.linalg.matmul(tensor,
                                                      tensor, transpose_b=True)
        distances = tf.math.maximum(distances, 0.0)
        return distances

@tf.keras.utils.register_keras_serializable(package="Similarity")
class EuclideanDistance(Distance):

    def __init__(self, name: str = None):
        """Compute pairwise Euclidean distances"""
        name = name if name else 'euclidean'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        x_rs = tf.reshape(embeddings, shape=[tf.shape(embeddings)[0], -1])
        squared_norm = tf.math.square(x_rs)
        squared_norm = tf.math.reduce_sum(squared_norm,
                                          axis=1,
                                          keepdims=True)

        distances: FloatTensor= 2.0 * tf.linalg.matmul(embeddings,
                                                       embeddings,
                                                       transpose_b=True)
        distances = squared_norm - distances + tf.transpose(squared_norm)

        # Avoid NaN and inf gradients when back propagating through the sqrt.
        # values smaller than 1e-18 produce inf for the gradient, and 0.0 produces NaN.
        # All values smaller than 1e-13 should produce a gradient of 1.0.
        dist_mask = tf.math.greater_equal(distances, 1e-18)
        distances = tf.math.maximum(distances, 1e-18)
        distances = tf.math.sqrt(distances) * tf.cast(dist_mask, tf.float32)

        return distances

@tf.keras.utils.register_keras_serializable(package="Similarity")
class ManhattanDistance(Distance):

    def __init__(self, name: str = None):
        """Compute pairwise Manhattan distances"""
        name = name if name else 'manhattan'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        x_rs = tf.reshape(embeddings, shape=[tf.shape(embeddings)[0], -1])
        deltas = tf.expand_dims(x_rs, axis=1) - tf.expand_dims(x_rs, axis=0)
        distances: FloatTensor = tf.norm(deltas, 1, axis=2)
        return distances


def distance_canonicalizer(distance: Union[Distance, str]) -> Distance:
    """Normalize user requested distance to its matching Distance object.

    Args:
        distance: Requested distance either by name or by object

    Returns:
        Distance: Requested object name.
    """
    mapping = {
        'cosine': 'cosine',
        'euclidean': 'euclidean',
        'l2': 'euclidean',
        'l1': 'manhattan',
    }

    if isinstance(distance, str):
        distance_name = distance.lower().strip()
        if distance_name in mapping:
            distance_name = mapping[distance_name]
        else:
            raise ValueError('Metric not supported by the framework')

        # instanciating
        if distance_name == 'cosine':
            return CosineDistance()
        elif distance_name == 'euclidean':
            return EuclideanDistance()
        elif distance_name == 'manhattan':
            return ManhattanDistance()

    elif isinstance(distance, Distance):
        # user supplied distance function
        return distance

    raise ValueError('Unknown distance: must either be a MetricDistance\
                          or a known distance function')
