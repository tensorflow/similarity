"Test that the example in the readme work correctly"
from tensorflow.keras import layers

from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler
from tensorflow_similarity.visualization import viz_neigbors_imgs


def test_readme_minimal():
    # Data sampler that generates balanced batches from MNIST dataset
    sampler = TFDatasetMultiShotMemorySampler(
        dataset_name='mnist',
        classes_per_batch=10
    )

    # Build a Similarity model using standard Keras layers
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.experimental.preprocessing.Rescaling(1/255)(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = MetricEmbedding(64)(x)

    # Build a specialized Similarity model
    model = SimilarityModel(inputs, outputs)

    # Train Similarity model using contrastive loss
    model.compile('adam', loss=MultiSimilarityLoss())
    model.fit(sampler, epochs=5)

    # Index 100 embedded MNIST examples to make them searchable
    sx, sy = sampler.get_slice(0, 100)
    model.index(x=sx, y=sy, data=sx)

    # Find the top 5 most similar indexed MNIST examples for a given example
    qx, qy = sampler.get_slice(3713, 1)
    nns = model.single_lookup(qx[0])

    # Visualize the query example and its top 5 neighbors
    viz_neigbors_imgs(qx[0], qy[0], nns)
