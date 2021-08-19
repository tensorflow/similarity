# TensorFlow Similarity: Metric Learning for Humans

TensorFlow Similarity is a [TensorFLow](https://tensorflow.org) library for 
[low supervision, metric (or "similarity") learning](https://en.wikipedia.org/wiki/Similarity_learning).

TensorFlow Similarity is still in beta.


## Introduction

Tensorflow Similarity offers state-of-the-art algorithms for metric learning
and all the necessary components to research, train, evaluate and serve
similarity-based models.

With TensorFlow Similarity you can train and serve models that find similar items
(for example, images) in a large corpus of examples.

In future releases, you will be able to use TensorFlow Similarity to perform
semi-supervised or self-supervised training, which can improve the training
on sparsely labeled corpora.

### Supervised models

Metric learning is different from traditional classification:

*Supervised models* learn to output metric embeddings (1D float tensor)
that exhibit the property that if two examples are close in the real world,
their embeddings will be close in the
projected [metric space](https://en.wikipedia.org/wiki/Metric_space).

(This can be compared to word embeddings: two words with similar meanings
will be close to each other in semantic space.)

Representing items by their metrics embeddings allows you to:

- build indexes that contains categories that were not seen during training
- add categories to the index without retraining
- train models on sparsely populated categories ("few shot learning")
- train models on sparsely labeled examples

#### Efficient

Retrieving similar items from the index is very efficient because
metric learning enables the use of [Approximate Nearest Neighboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
which is completed in sublinear time, rather than using the standard
[Nearest Neighboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search),
which takes quadratic time.

TensorFlow Similarity uses [NMSLIB](https://github.com/nmslib/nmslib),
which can find the closest items in a fraction of second,
even with the index containing over a million elements.

### Self-supervised models

**This is a work in progress.**

*Self-supervised contrastive models* help make classification models
more accurate by performing large-scale pretrainings that aim at learning
a consistent representation of the data, by "contrasting" different representation of
the same example generated via data augmentation and/or contrasting the
representation of different examples to separate then. Then the model is
fine-tuned on the few labeled examples like any classification model.

## What's new

- [Aug 21]: Interactive embedding `projector()` added. See this [notebook](examples/supervized_visualization/)
- [Aug 21]: `CircleLoss()` added
- [Aug 21]: `PNLoss()` added.
- [Aug 21]: `MultiSimilarityLoss()` added.


For previous changes - see [the release changelog](.releases.md)

## Getting Started

### Installation

Use pip to install the library

```python
pip install tensorflow_similarity
```

### Documentation

The detailed and narrated notebooks are a good way to get started
with TensorFlow Similarity. There is likely to be one that is similar to
your data or your problem (if not, let us know). You can start working with
the examples immediately in Google Colab by clicking the Google colab icon.

For more information about specific functions, you can [check the API documentation -- FIXME]()


## Example: MNIST similarity

### Preparing data

```python
from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler
sampler = TFDatasetMultiShotMemorySampler(dataset_name='mnist', class_per_batch=10)
```

### Building a Similarity model

```python
from tensorflow.keras import layers
from tensorflow_similarity.models import SimilarityModel
inputs = layers.Input(shape=(spl.x[0].shape))
x = layers.experimental.preprocessing.Rescaling(1/255)(inputs)
x = layers.Conv2D(32, 7, activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64)(x)
model = SimilarityModel(inputs, x)
```

### Training model via contrastive learning

```python
from tensorflow_similarity.losses import TripletLoss
# using Tripletloss to project in metric space
tloss = TripletLoss()
model.compile('adam', loss=tloss)
model.fit(sampler, epochs=5)
```

### Building images index and querying it

```python
from tensorflow_similarity.visualization import viz_neigbors_imgs

# index emneddings for fast retrivial via ANN
model.index(x=sampler.x[:100], y=sampler.y[:100], data=sampler.x[:100])

# Lookup examples nearest indexed images
nns = model.single_lookup(sampler.x[3713])

# visualize results result
viz_neigbors_imgs(sampler.x[3713], sampler.y[3713], nns)
```


## Supported Algorithms

- Triplet Loss 
- PN Loss 
- Multi Loss
- Circle Loss


## Package components

![TensorFlow Similarity Overview](api/images/tfsim_overview.png)

TensorFlow Similiarity, as visible in the diagram above, offers the following
components to help research, train, evaluate and serve metric models:

- **`SimilarityModel()`**: This class subclasses the `tf.keras.model` class and extends it with additional properties that are useful for metric learning. For example it adds the methods:
  1. `index()`: Enables indexing of the embedding
  2. `lookup()`: Takes samples, calls predict(), and searches for neighbors within the index.

- **`MetricLoss()`**:  This virtual class, that extends the `tf.keras.Loss` class, is the base class from which Metric losses are derived from. This subclassing ensures proper error checking; that is, it ensures the user is using a loss metric to train the models, perform better static analysis, and enforces additional constraints such as having a distance function that is supported by the index. Additionally, Metric losses make use of the fully tested and highly optimized pairwise distances functions provided by TensorFlow Similarity that are available under the `Distances.*` classes.

- **`Samplers()`**: Samplers are meant to ensure that each batch has at least n (with n >=2) examples of each class, as losses such as TripletLoss can’t work properly if this condition is not met. TensorFlow Similarity offers an in-memory sampler for small dataset and a `tf.data.TFRecordDataset` for large scales one.

- **`Indexer()`**: The Indexer and its sub-components are meant to index known embeddings alongside their metadata. The embedding metadata is stored within `Table()`, while the `Matcher()` is used to perform [fast approximate neighboor searches](https://en.wikipedia.org/wiki/Nearest_neighbor_search) that are meant to quickly retrieve the indexed elements that are the closest to the embeddings supplied in the `lookup()` and `single_lookup()` function.
The `Evaluator()` component is used to compute EvalMetrics() on the specific index for evaluation and calibration purpose.

The default `Index()` sub-compoments run in-memory and are optimized to be used in interactive settings such as jupyter notebooks, colab, and metric computation during training (e.g using the `EvalCallback()` provided). Index are serialized as part of `model.save()` so you can reload them via `model.index_load()` for serving purpose or futher training / evaluation.

The default implementation can scale up to medium deployment (1M-10M+ points) easily provided the computers used have enough memory. For very large scale deployement you will need to sublcass the compoments to match your own architetctue. See FIXME colab to see how to deploy TensorFlow Similarity in production.


For more information about a given component head to the [API documentation](FIXME) or read [the TensorFlow Similarity paper](FIXME).


## Citing

Please cite this reference if you use any part of TensorFlow similarity
in your research:

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
