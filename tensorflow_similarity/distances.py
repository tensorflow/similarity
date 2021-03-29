from abc import ABC
from typing import Union
import tensorflow as tf
from .types import FloatTensor


class Distance(ABC):
    def __init__(self, name: str):
        self.name = name

    def call(self, embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch.

        Args:
            embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """

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
    """Compute pairwise cosine distances between embeddings.

    The [Cosine Distance](https://en.wikipedia.org/wiki/Cosine_similarity) is
    an angular distance that varies from 0 (similar) to 1 (dissimilar).
    """
    def __init__(self, name: str = None):
        "Init Cosine distance"
        name = name if name else 'cosine'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch of embeddings.

        Args:
            embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
        tensor = tf.nn.l2_normalize(embeddings, axis=1)
        distances: FloatTensor = 1 - tf.linalg.matmul(tensor,
                                                      tensor, transpose_b=True)
        distances = tf.math.maximum(distances, 0.0)
        return distances


@tf.keras.utils.register_keras_serializable(package="Similarity")
class EuclideanDistance(Distance):
    """Compute pairwise euclidean distances between embeddings.

    The [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
    is the standard distance to measure the line segment between two embeddings
    in the Cartesian point. The larger the distance the more dissimilar
    the embeddings are.

    **Alias**: L2 Norm, Pythagorean
    """

    def __init__(self, name: str = None):
        "Init Euclidean distance"
        name = name if name else 'euclidean'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch of embeddings.

        Args:
            embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
        squared_norm = tf.math.square(embeddings)
        squared_norm = tf.math.reduce_sum(squared_norm,
                                          axis=1,
                                          keepdims=True)

        distances: FloatTensor = 2.0 * tf.linalg.matmul(embeddings,
                                                        embeddings,
                                                        transpose_b=True)
        distances = squared_norm - distances + tf.transpose(squared_norm)

        # Avoid NaN and inf gradients when back propagating through the sqrt.
        # values smaller than 1e-18 produce inf for the gradient, and 0.0
        # produces NaN. All values smaller than 1e-13 should produce a gradient
        # of 1.0.
        dist_mask = tf.math.greater_equal(distances, 1e-18)
        distances = tf.math.maximum(distances, 1e-18)
        distances = tf.math.sqrt(distances) * tf.cast(dist_mask, tf.float32)

        return distances


@tf.keras.utils.register_keras_serializable(package="Similarity")
class ManhattanDistance(Distance):
    """Compute pairwise Manhattan distances between embeddings.

    The [Manhattan Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
    is the sum of the lengths of the projections of the line segment between
    two embeddings onto the Cartesian axes. The larger the distance the more
    dissimilar the embeddings are.

    **Alias**: L1 Norm, taxicab
    """

    def __init__(self, name: str = None):
        "Init Manhattan distance"
        name = name if name else 'manhattan'
        super().__init__(name)

    @tf.function
    def call(self, embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch of embeddings.

        Args:
            embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
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
        'pythagorean': 'euclidean',
        'l2': 'euclidean',
        'l1': 'manhattan',
        'taxicab': 'manhattan',
    }

    if isinstance(distance, str):
        distance_name = distance.lower().strip()
        if distance_name in mapping:
            distance_name = mapping[distance_name]
        else:
            raise ValueError('Metric not supported by the framework')

        # instantiating
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
