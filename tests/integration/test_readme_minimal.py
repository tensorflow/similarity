"Test that the example in the readme work correctly"
import os

import pytest
from tensorflow.keras import layers

from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler


def test_readme_minimal():
    """This should be nearly identical to the README code."""
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
    nns = model.single_lookup(qx[0])  # noqa

    # ! don't add viz its block the test in certain env.
    # Visualize the query example and its top 5 neighbors
    # viz_neigbors_imgs(qx[0], qy[0], nns)


@pytest.fixture
def readme_path(request):
    """Helper to load README relative to the test file."""
    # README path needs to be relative to the test.
    test_path = os.path.dirname(os.path.realpath(request.module.__file__))
    return os.path.join(test_path, '..', '..', 'README.md')


def test_readme_text_directly(readme_path):
    """Quick and dirty test of the README.md code snippets."""
    code = []
    code_block = False

    with open(readme_path, 'r') as f:
        for line in f:
            if line.endswith("```\n"):
                code_block = False

            # Add all code lines except for the viz function.
            if code_block and not line.startswith('viz_neighbors_imgs'):
                code.append(line)

            if line.startswith("```python"):
                code_block = True

    exec(('\n').join(code))
