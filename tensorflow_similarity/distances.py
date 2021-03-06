from abc import abstractmethod, ABC
from typing import Union
import tensorflow as tf
from .types import FloatTensor


class Distance(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def call(self, embeddings: FloatTensor, axis: int = 1) -> FloatTensor:
        """Compute distance"""

    def __call__(self, embeddings: FloatTensor, axis: int = 1):
        return self.call(embeddings, axis)

    def __str__(self) -> str:
        return self.name

    def get_config(self):
        return {
            "name": self.name
        }

@tf.keras.utils.register_keras_serializable(package="Similarity")
class CosineDistance(Distance):

    def __init__(self, name: str = None):
        """Compute pairwises cosine distances"""
        name = name if name else 'cosine'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor, axis: int = 1) -> FloatTensor:
        tensor = tf.nn.l2_normalize(embeddings, axis=axis)
        distances: FloatTensor = 1 - tf.linalg.matmul(tensor,
                                                      tensor, transpose_b=True)
        distances = tf.math.maximum(distances, 0.0)
        return distances

@tf.keras.utils.register_keras_serializable(package="Similarity")
class EuclidianDistance(Distance):

    def __init__(self, name: str = None):
        """Compute pairwises Euclidian distances"""
        name = name if name else 'euclidian'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor, axis: int = 1) -> FloatTensor:
        squared_norm = tf.math.square(embeddings)
        squared_norm = tf.math.reduce_sum(squared_norm,
                                          axis=axis,
                                          keepdims=True)

        distances: FloatTensor = 2.0 * tf.linalg.matmul(embeddings,
                                                        embeddings,
                                                        transpose_b=True)
        distances = squared_norm - distances + tf.transpose(squared_norm)

        # Avoid NaN gradients when back propegating through the sqrt.
        distances = tf.math.maximum(distances, 1e-16)
        distances = tf.math.sqrt(distances)

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
        'euclidian': 'euclidian',
        'l2': 'euclidian'
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
        elif distance_name == 'euclidian':
            return EuclidianDistance()

    elif isinstance(distance, Distance):
        # user supplied distance function
        return distance

    raise ValueError('Unknown distance: must either be a MetricDistance\
                          or a known distance function')
