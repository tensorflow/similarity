# TensorFlow Similarity: Metric Learning for Humans

TensorFlow Similarity is a [TensorFLow](https://tensorflow.org) library for [similarity learning](https://en.wikipedia.org/wiki/Similarity_learning) also known as metric learning and contrastive learning.

TensorFlow Similarity is still in beta.

## Introduction

Tensorflow Similarity offers state-of-the-art algorithms for metric learning and all the necessary components to research, train, evaluate, and serve similarity-based models.

With TensorFlow Similarity you can train and serve models that find similar items
(such as images) in a large corpus of examples.

TODO(elie): Add matching cats and dogs

Metric learning is different from traditional classification as it's objective is different. The model learns to minimize the distance between similar examples and maximize the distance between different examples, in a supervised or self-supervised fashion. Either way, TensorFlow Similarity provides the necessary losses, metrics, samplers, visualizers, and indexing sub-system to make this easy and fast. 

To learn more about the benefits of using similarity training, you can check out the blog post.

Currently, TensorFlow Similarity supports supervised training. In future releases, it will support semi-supervised and self-supervised training. 

## What's new

- [Aug 21]: Interactive embedding `projector()` added. See this [notebook](examples/supervized_visualization/)
- [Aug 21]: `CircleLoss()` added
- [Aug 21]: `PNLoss()` added.
- [Aug 21]: `MultiSimilarityLoss()` added.

For previous changes - see [the release changelog](./releases.md)

## Getting Started

### Installation

Use pip to install the library

```python
pip install tensorflow_similarity
```

### Documentation

The detailed and narrated notebooks are a good way to get started with TensorFlow Similarity. There is likely to be one that is similar to your data or your problem (if not, let us know). You can start working with the examples immediately in Google Colab by clicking the Google Colab icon.

For more information about specific functions, you can [check the API documentation](api/)

### Minimal Example: MNIST similarity

Here is a bare bones example demonstrating how to train a TensorFlow Similarity model on the MNIST data. This example illustrates some of the main components provided by TensorFlow Similarity and how they fit together. Please refer to the hello_world notebook for a more detailed introduction.

### Preparing data

TensorFlow Similarity provides data samplers that ensure that:
- Support restricting the batches to a subset of the classes present in the dataset.
- Limit the number of examples in the class.
- Ensure that batches contain at least N examples of each class present in the batch, as required by the similarity losses.

In this example, we are using the multi shot sampler that pulls directly from the TensorFlow dataset catalog, without using any class filtering. See API documentation for full the list of samplers, and the narrated notebooks for concrete examples.

```python
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler

# Data sampler that generates balanced batches from MNIST dataset
sampler = TFDatasetMultiShotMemorySampler(dataset_name='mnist', class_per_batch=10)
```

### Building a Similarity model

Building a TensorFlow Similarity model is similar to building a standard Keras model, except the output layer is usually a `MetricEmbedding()` layer that enforces L2 normalization and the model is instantiated as a specialized subclass `SimilarityModel()` that supports additional functionality.

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

The model requires the use of special TensorFlow Similarity losses that construct the triplets from each batch of examples.

```python
from tensorflow_similarity.losses import MultiSimilarityLoss

# Train Similarity model using contrastive loss
model.compile('adam', loss=MultiSimilarityLoss())
model.fit(sampler, epochs=5)
```

### Building images index and querying it

Once the model is trained, embedded examples can be added to the index. Users can then use an embedded query to search the indexed examples for the K most similar items.

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

- Triplet Loss 
- PN Loss 
- Multi Loss
- Circle Loss

## Package components

![TensorFlow Similarity Overview](assets/images/tfsim_overview.png)

TensorFlow Similiarity, as visible in the diagram above, offers the following
components to help research, train, evaluate and serve metric models:

- **`SimilarityModel()`**: This class subclasses the `tf.keras.model` class and extends it with additional properties that are useful for metric learning. For example it adds the methods:
  1. `index()`: Enables indexing of the embedding
  2. `lookup()`: Takes samples, calls predict(), and searches for neighbors within the index.
  3. `calibrate()`: Calibrates the model's index search thresholds using a calibration metric and a test dataset.

- **`MetricLoss()`**:  This virtual class, that extends the `tf.keras.Loss` class, is the base class from which Metric losses are derived. This sub-classing ensures proper error checking; that is, it ensures the user is using a loss metric to train the models, performs better static analysis, and enforces additional constraints such as having a distance function that is supported by the index. Additionally, Metric losses make use of the fully tested and highly optimized pairwise distances functions provided by TensorFlow Similarity that are available under the `Distances.*` classes.

- **`Samplers()`**: Samplers are meant to ensure that each batch has at least n (with n >=2) examples of each class, as losses such as TripletLoss can’t work properly if this condition is not met. TensorFlow Similarity offers an in-memory sampler for small dataset and a `tf.data.TFRecordDataset` for large scales one.

- **`Indexer()`**: The Indexer and its sub-components are meant to index known embeddings alongside their metadata. The embedding metadata is stored within `Table()`, while the `Matcher()` is used to perform [fast approximate neighbor searches](https://en.wikipedia.org/wiki/Nearest_neighbor_search) that are meant to quickly retrieve the indexed elements that are the closest to the embeddings supplied in the `lookup()` and `single_lookup()` function.

The default `Index()` sub-compoments run in-memory and are optimized to be used in interactive settings such as Jupyter notebooks, Colab, and metric computation during training (e.g using the `EvalCallback()` provided). Index are serialized as part of `model.save()` so you can reload them via `model.index_load()` for serving purpose or further training / evaluation.

The default implementation can scale up to medium deployment (1M-10M+ points) easily, provided the computers have enough memory. For very large scale deployments you will need to sublcass the compoments to match your own architetctue. See FIXME colab to see how to deploy TensorFlow Similarity in production.

For more information about a given component head to the [API documentation](api/) or read [the TensorFlow Similarity paper](FIXME).


## Citing

Please cite this reference if you use any part of TensorFlow similarity in your research:

```bibtex
@article{EBSIM21,
  title={TensorFlow Similarity: A Usuable, High-Performance Metric Learning Library},
  author={Elie Bursztein, James Long, Shun Lim, Owen Vallis, Francois Chollet},
  journal={Fixme},
  year={2021}
}
```

## Disclaimer

This is not an official Google product.
