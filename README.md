# TensorFlow Similarity: Metric Learning for Humans

TensorFlow Similarity is a [TensorFLow](https://tensorflow.org) library for [similarity learning](https://en.wikipedia.org/wiki/Similarity_learning) also known as metric learning and contrastive learning.

TensorFlow Similarity is still in beta.

## Introduction

Tensorflow Similarity offers state-of-the-art algorithms for metric learning and all the necessary components to research, train, evaluate, and serve similarity-based models.

![Example of nearest neighbors search performed on the embedding generated by a similarity model trained on the Oxford IIIT Pet Dataset.](assets/images/similar-cats-and-dogs.jpg)

With TensorFlow Similarity you can train and serve models that find similar items (such as images) in a large corpus of examples. For example, as visible above, you can train a similarity model to find and cluster similar looking images of cats and dogs from the [Oxford IIIT Pet Dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) by only training on a few classes. To train your own similarity model see this [notebook](examples/supervised_visualization.ipynb).

Metric learning is different from traditional classification as it's objective is different. The model learns to minimize the distance between similar examples and maximize the distance between dissimilar examples, in a supervised or self-supervised fashion. Either way, TensorFlow Similarity provides the necessary losses, metrics, samplers, visualizers, and indexing sub-system to make this quick and easy.

**Currently, TensorFlow Similarity supports supervised training.** In future releases, it will support semi-supervised and self-supervised training.

To learn more about the benefits of using similarity training, you can check out the blog post.

## What's new

- [Aug 21]: Interactive embedding `projector()` added. See this [notebook](examples/supervised_visualization.ipynb)
- [Aug 21]: [`CircleLoss()`](api/TFSimilarity/losses/CircleLoss.md) added
- [Aug 21]: [`PNLoss()`](api/TFSimilarity/losses/PNLoss.md) added.
- [Aug 21]: [`MultiSimilarityLoss()`](api/TFSimilarity/losses/MultiSimilarityLoss.md) added.

For previous changes - see [the release changelog](./releases.md)

## Getting Started

### Installation

Use pip to install the library

```python
pip install tensorflow_similarity
```

### Documentation

The detailed and narrated [notebooks](examples/) are a good way to get started with TensorFlow Similarity. There is likely to be one that is similar to your data or your problem (if not, let us know). You can start working with the examples immediately in Google Colab by clicking the Google Colab icon.

For more information about specific functions, you can [check the API documentation](api/)

For contributing to the project please check out the [contribution guidelines](CONTRIBUTING.md)

### Minimal Example: MNIST similarity

Here is a bare bones example demonstrating how to train a TensorFlow Similarity model on the MNIST data. This example illustrates some of the main components provided by TensorFlow Similarity and how they fit together. Please refer to the [hello_world notebook](examples/supervised_hello_world.ipynb) for a more detailed introduction.

### Preparing data

TensorFlow Similarity provides [data samplers](api/TFSimilarity/samplers/), for various dataset types, that balance the batches to ensure smoother training.
In this example, we are using the multi-shot sampler that integrate directly from the TensorFlow dataset catalog.

```python
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler

# Data sampler that generates balanced batches from MNIST dataset
sampler = TFDatasetMultiShotMemorySampler(dataset_name='mnist', class_per_batch=10)
```

### Building a Similarity model

Building a TensorFlow Similarity model is similar to building a standard Keras model, except the output layer is usually a [`MetricEmbedding()`](api/TFSimilarity/layers/) layer that enforces L2 normalization and the model is instantiated as a specialized subclass [`SimilarityModel()`](api/TFSimilarity/models/SimilarityModel.md) that supports additional functionality.

```python
from tensorflow.keras import layers
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.models import SimilarityModel

# Build a Similarity model using standard Keras layers
inputs = layers.Input(shape=(28, 28, 1))
x = layers.experimental.preprocessing.Rescaling(1/255)(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = MetricEmbedding(64)(x)

# Build a specialized Similarity model
model = SimilarityModel(inputs, outputs)
```

### Training model via contrastive learning

To output a metric embedding, that are searchable via approximate nearest neighbor search, the model needs to be trained using a similarity loss. Here we are using the `MultiSimilarityLoss()`, which is one of the most efficient loss functions.

```python
from tensorflow_similarity.losses import MultiSimilarityLoss

# Train Similarity model using contrastive loss
model.compile('adam', loss=MultiSimilarityLoss())
model.fit(sampler, epochs=5)
```

### Building images index and querying it

Once the model is trained, reference examples must indexed via the model index API to be searchable. After indexing, you can use the model lookup API to search the index for the K most similar items.

```python
from tensorflow_similarity.visualization import viz_neigbors_imgs

# Index 100 embedded MNIST examples to make them searchable
model.index(x=sampler.x[:100], y=sampler.y[:100], data=sampler.x[:100])

# Find the top 5 most similar indexed MNIST examples for a given example
nns = model.single_lookup(sampler.x[3713])

# Visualize the query example and its top 5 neighbors
viz_neigbors_imgs(sampler.x[3713], sampler.y[3713], nns)
```

## Supported Algorithms

### Supervised Losses

- Triplet Loss
- PN Loss
- Multi Sim Loss
- Circle Loss

### Metrics

Tensorflow Similarity offers many of the most common metrics used for [classification](api/TFSimilarity/classification_metrics/) and [retrieval](api/TFSimilarity/retrieval_metrics/) evaluation. Including:

| Name | Type | Description |
| ---- | ---- | ----------- |
| Precision | Classification | |
| Recall | Classification | |
| F1 Score | Classification | |
| Recall@K | Retrieval | |
| Binary NDCG | Retrieval | |

## Citing

Please cite this reference if you use any part of TensorFlow similarity in your research:

```bibtex
@article{EBSIM21,
  title={TensorFlow Similarity: A Usuable, High-Performance Metric Learning Library},
  author={Elie Bursztein, James Long, Shun Lin, Owen Vallis, Francois Chollet},
  journal={Fixme},
  year={2021}
}
```

## Disclaimer

This is not an official Google product.
