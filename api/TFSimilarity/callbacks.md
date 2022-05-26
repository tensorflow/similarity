# Module: TFSimilarity.callbacks





Specialized callbacks that track similarity metrics during training



## Classes

- [`class Callback`](../TFSimilarity/callbacks/Callback.md): Abstract base class used to build new callbacks.

- [`class ClassificationMatch`](../TFSimilarity/callbacks/ClassificationMatch.md): Abstract base class for defining the classification matching strategy.

- [`class ClassificationMetric`](../TFSimilarity/callbacks/ClassificationMetric.md): Abstract base class for computing classification metrics.

- [`class EvalCallback`](../TFSimilarity/callbacks/EvalCallback.md): Epoch end evaluation callback that build a test index and evaluate

- [`class Evaluator`](../TFSimilarity/callbacks/Evaluator.md): Evaluates search index performance and calibrates it.

- [`class FloatTensor`](../TFSimilarity/callbacks/FloatTensor.md): Float tensor 

- [`class IntTensor`](../TFSimilarity/callbacks/IntTensor.md): Integer tensor

- [`class MemoryEvaluator`](../TFSimilarity/callbacks/MemoryEvaluator.md): In memory index performance evaluation and classification.

- [`class Path`](../TFSimilarity/callbacks/Path.md): PurePath subclass that can make system calls.

- [`class SimilarityModel`](../TFSimilarity/callbacks/SimilarityModel.md): Specialized Keras.Model which implement the core features needed for

- [`class Tensor`](../TFSimilarity/callbacks/Tensor.md): A `tf.Tensor` represents a multidimensional array of elements.

## Functions

- [`SplitValidationLoss(...)`](../TFSimilarity/callbacks/SplitValidationLoss.md): Creates the validation callbacks.

- [`make_classification_metric(...)`](../TFSimilarity/callbacks/make_classification_metric.md): Convert classification metric from str name to object if needed.

- [`unpack_lookup_distances(...)`](../TFSimilarity/callbacks/unpack_lookup_distances.md)

- [`unpack_lookup_labels(...)`](../TFSimilarity/callbacks/unpack_lookup_labels.md)

- [`unpack_results(...)`](../TFSimilarity/callbacks/unpack_results.md): Updates logs, writes summary, and returns list of strings of

