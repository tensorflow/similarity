# TensorFlow Similarity: Metric Learning for Humans

TensorFlow Similarity is a [TensorFLow](https://tensorflow.org) package focused on making metric learning easy. Whether you are looking to train and serve models meant to find similar items (images/text/sounds) or apply semi-supervised contrastive techniques to boost models accuracy TensorFlow Similarity can help you start quickly and scale up efficiently. Its set of well-tested composable components that follows Keras best practices are meant to be seamlessly integrated into your TensorFlow workflows to get you results faster whether you are doing research or building products.



## Getting Started

### Installation

Use pip to install the library

```python
pip install tensorflow_similarity
```

### Tutorials

We created a few tutorials that highlights TF similarity core features to help you getting started:

| Tutorial | Learning Type | Description |
| ------ | :-----: | ---------- |
| [Hello World](notebooks/hello_world.ipynb) | Supervised | Train and use an image similarity model to find related MNIST digits |
| TODO  | Semi-supervised| Train a semi-supervised model on CIFAR10 using contrastive learning |
| [Stanford Dogs](notebooks/supervised_advanced.ipynb) | Supervised | Train an image similarity classifiers on the Stanford Dogs dataset using  some of the TF Similarity advanced features such as the TFRecordSampler and Evaluation Callbacks|

## Available components

![TensorFlow Similarity Overview](documentations/images/tfsim_overview.png)

As visible in the diagram above, TensorFlow Similiarity offers the following components:

- **`SimilarityModel()`**: This class subclasses the `tf.keras.model` class and extends it with additional properties that are useful for metric learning. For example it adds the methods:
  1. `index()`: Enables indexing of the embedding
  2. `lookup()`: Takes samples, calls predict(), and searches for neighbors within the index.

- **`MetricLoss()`**:  This virtual class, that extends the `tf.keras.Loss` class, is the base class from which Metric losses are derived from. This subclassing ensures proper error checking, i.e., ensures the user is using a loss metric to train the models, perform better static analysis, and enforces additional constraints such as having a distance function that is supported by the index. Additionally, Metric losses make use of the fully tested and highly optimized pairwise distances functions provided by TF Similarity that are available under the `Distances.*` classes.

- **`Samplers()`**: Samplers are meant to ensure that each batch has at least n (with n >=2) examples of each class, as losses such as TripletLoss can’t work properly if this condition is not met. TF similarity offers an in-memory sampler for small dataset and a TFRecordDatasets for large scales one.

- **`Indexer()`**: The Indexer and its sub-component are meant to index known embeddings alongside their metadata. The embedding metadata is stored within `Table()`, while the `Matcher()` is used to perform [fast approximate neighboor searches](https://en.wikipedia.org/wiki/Nearest_neighbor_search) that are meant to quickly retrieve the indexed elements that are the closest to the embeddings supplied in the `lookup()` and `single_lookup()` function.
The `Evaluator()` component is used to compute EvalMetrics() on the specific index for evaluation and calibration purpose.

The default `Index()` sub-compoments run in-memory and are optimized to be used in interactive settings such as jupyter notebooks, colab, and metric computation during training (e.g using the `EvalCallback()` provided). Index are serialized as part of `model.save()` so you can reload them via `model.index_load()` for serving purpose or futher training / evaluation.

The default implementation can scale up to medium deployement (1M-10M+ points) easily provided the computers used have enough memory. For very large scale deployement you will need to sublcass the compoments to match your own architetctue. See FIXME colab to see how to deploy TF simialrity in production.


For more information about a given component head to the [API documentation](FIXME) or read [the TensorFlow Similarity paper](FIXME).


## Reference

Please cite this reference if you use TensorFlow similarity in your research

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
