import pytest

from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler


def test_basic():
    dataset_name = "mnist"
    sampler = TFDatasetMultiShotMemorySampler(dataset_name=dataset_name, classes_per_batch=10)
    batch = sampler.generate_batch(42)
    assert batch[0].shape == (20, 28, 28, 1)


def test_wrong_key():
    dataset_name = "mnist"

    # X
    with pytest.raises(ValueError):
        TFDatasetMultiShotMemorySampler(dataset_name=dataset_name, classes_per_batch=4, x_key="error")
    # Y
    with pytest.raises(ValueError):
        TFDatasetMultiShotMemorySampler(dataset_name=dataset_name, classes_per_batch=4, y_key="error")
