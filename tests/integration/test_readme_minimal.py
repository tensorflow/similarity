"Test that the example in the readme work correctly"
import os

import pytest
from tensorflow.keras import layers

from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler


@pytest.fixture
def readme_path(request):
    """Helper to load README relative to the test file."""
    # README path needs to be relative to the test.
    test_path = os.path.dirname(os.path.realpath(request.module.__file__))
    return os.path.join(test_path, "..", "..", "README.md")


def test_readme_text_directly(readme_path):
    """Quick and dirty test of the README.md code snippets."""
    code = []
    code_block = False

    with open(readme_path, "r") as f:
        for line in f:
            if line.endswith("```\n"):
                code_block = False

            # Add all code lines except for the viz function.
            if code_block and not line.startswith("viz_neighbors_imgs"):
                code.append(line)

            if line.startswith("```python"):
                code_block = True

    exec(("\n").join(code))
