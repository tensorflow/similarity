# TensorFlow Similarity: Metric Learning for Humans

TensorFlow Similarity is a [TensorFlow](https://tensorflow.org) library for [similarity learning](https://en.wikipedia.org/wiki/Similarity_learning) which includes techniques such as self-supervised learning, metric learning, similarity learning, and contrastive learning. TensorFlow Similarity is still in beta and we may push breaking changes.

## Introduction

Tensorflow Similarity offers state-of-the-art algorithms for metric learning along with all the necessary components to research, train, evaluate, and serve similarity and contrastive based models. These components include models, losses, metrics, samplers, visualizers, and indexing subsystems to make this quick and easy.

![Example of nearest neighbors search performed on the embedding generated by a similarity model trained on the Oxford IIIT Pet Dataset.](https://raw.githubusercontent.com/tensorflow/similarity/master/assets/images/similar-cats-and-dogs.jpg)

With Tensorflow Similarity you can train two main types of models:

1. **Self-supervised models**: Used to learn general data representations on unlabeled data to boost the accuracy of downstream tasks where you have few labels. For example, you can pre-train a model on a large number of unlabled images using one of the supported contrastive methods supported by TensorFlow Similarity, and then fine-tune it on a small labeled dataset to achieve higher accuracy. To get started training your own self-supervised model see this [notebook](examples/unsupervised_hello_world.ipynb).

2. **Similarity models**: Output embeddings that allow you to find and cluster similar examples such as images representing the same object within a large corpus of examples. For instance, as visible above, you can train a similarity model to find and cluster similar looking, unseen cat and dog images from the [Oxford IIIT Pet Dataset](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) while only training on a few of the dataset classes. To get started training your own similarity model see this [notebook](examples/supervised/visualization.ipynb).

## What's new

- [Jan 2022]: 0.15 self-supervised release
    * Added support for self-supervised contrastive learning. Including SimCLR, SimSiam, and Barlow Twins. Checkout the in-depth [hello world notebook](examples/unsupervised_hello_world.ipynb) to get started.
    * Soft Nearest Neighbor Loss added thanks to [Abhishar Sinha](https://github.com/abhisharsinha)
    * Added GenerlizedMeanPooling2D support that improves similarity matching accuracy over GlobalMeanPooling2D.
    * Numerous speed optimizations and general bug fixes.
- [Dec 2021]: 
    * Sampler speed optimizations and general bug fixes.
- [Oct 2021]: 0.14 similarity release
    * 0.14 is out which includes various speed improvements and post initial release bug fixes.
    * Added `Samplers.*` IO [notebook](examples/sampler_io_cookbook.ipynb) detailing how to efficiently sample your data for successful training.
    

For previous changes and more details - see [the changelog](./releases.md)

## Getting Started

### Installation

Use pip to install the library.

**NOTE**: The Tensorflow extra_require key can be omitted if you already have tensorflow>=2.4 installed.

```shell
pip install --upgrade-strategy=only-if-needed tensorflow_similarity[tensorflow] 
```

### Documentation

The detailed and narrated [notebooks](examples/) are a good way to get started with TensorFlow Similarity. There is likely to be one that is similar to your data or your problem (if not, let us know). You can start working with the examples immediately in Google Colab by clicking the Google Colab icon.

For more information about specific functions, you can [check the API documentation](api/)

For contributing to the project please check out the [contribution guidelines](CONTRIBUTING.md)

### Minimal Example: MNIST similarity
<details>
   <summary> Click to expand and see how to train a supervised similarity model on mnist using TF.Similarity</summary>

Here is a bare bones example demonstrating how to train a TensorFlow Similarity model on the MNIST data. This example illustrates some of the main components provided by TensorFlow Similarity and how they fit together. Please refer to the [hello_world notebook](examples/supervised_hello_world.ipynb) for a more detailed introduction.

### Preparing data

TensorFlow Similarity provides [data samplers](api/TFSimilarity/samplers/), for various dataset types, that balance the batches to ensure smoother training.
In this example, we are using the multi-shot sampler that integrates directly from the TensorFlow dataset catalog.

```python
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler

# Data sampler that generates balanced batches from MNIST dataset
sampler = TFDatasetMultiShotMemorySampler(dataset_name='mnist', classes_per_batch=10)
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

Once the model is trained, reference examples must be indexed via the model index API to be searchable. After indexing, you can use the model lookup API to search the index for the K most similar items.

```python
from tensorflow_similarity.visualization import viz_neigbors_imgs

# Index 100 embedded MNIST examples to make them searchable
sx, sy = sampler.get_slice(0,100)
model.index(x=sx, y=sy, data=sx)

# Find the top 5 most similar indexed MNIST examples for a given example
qx, qy = sampler.get_slice(3713, 1)
nns = model.single_lookup(qx[0])

# Visualize the query example and its top 5 neighbors
viz_neigbors_imgs(qx[0], qy[0], nns)
```
</details>

## Supported Algorithms

### Self-Supervised Models

- SimCLR 
- SimSiam
- Barlow Twins

### Supervised Losses

- Triplet Loss
- PN Loss
- Multi Sim Loss
- Circle Loss
- Soft Nearest Neighbor Loss

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
  title={TensorFlow Similarity: A Usable, High-Performance Metric Learning Library},
  author={Elie Bursztein, James Long, Shun Lin, Owen Vallis, Francois Chollet},
  journal={Fixme},
  year={2021}
}
```

## Disclaimer

This is not an official Google product.
