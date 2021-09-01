# Module: TFSimilarity.callbacks





Specialized callbacks that track similarity metrics during training



## Classes

- [`class Callback`](../TFSimilarity/callbacks/Callback.md): Abstract base class used to build new callbacks.

- [`class ClassificationMetric`](../TFSimilarity/callbacks/ClassificationMetric.md): Abstract base class for computing classification metrics.

- [`class EvalCallback`](../TFSimilarity/callbacks/EvalCallback.md): Epoch end evaluation callback that build a test index and evaluate

- [`class Evaluator`](../TFSimilarity/callbacks/Evaluator.md): Evaluates search index performance and calibrates it.

- [`class IntTensor`](../TFSimilarity/callbacks/IntTensor.md): Integer tensor

- [`class MemoryEvaluator`](../TFSimilarity/callbacks/MemoryEvaluator.md): In memory index performance evaluation and classification.

- [`class Path`](../TFSimilarity/callbacks/Path.md): PurePath subclass that can make system calls.

- [`class SimilarityModel`](../TFSimilarity/callbacks/SimilarityModel.md): Specialized Keras.Model which implement the core features needed for

- [`class SplitValidationLoss`](../TFSimilarity/callbacks/SplitValidationLoss.md): A split validation callback.

- [`class Tensor`](../TFSimilarity/callbacks/Tensor.md): A tensor is a multidimensional array of elements represented by a

## Functions

- [`make_classification_metric(...)`](../TFSimilarity/callbacks/make_classification_metric.md): Convert classification metric from str name to object if needed.

- [`unpack_lookup_distances(...)`](../TFSimilarity/callbacks/unpack_lookup_distances.md)

- [`unpack_lookup_labels(...)`](../TFSimilarity/callbacks/unpack_lookup_labels.md)

