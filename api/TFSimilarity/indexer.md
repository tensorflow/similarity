# Module: TFSimilarity.indexer





Index the embeddings infered by the model to allow distance based

sub-linear search

## Classes

- [`class CalibrationResults`](../TFSimilarity/indexer/CalibrationResults.md): Cutpoints and thresholds associated with a calibration.

- [`class ClassificationMatch`](../TFSimilarity/callbacks/ClassificationMatch.md): Abstract base class for defining the classification matching strategy.

- [`class ClassificationMetric`](../TFSimilarity/callbacks/ClassificationMetric.md): Abstract base class for computing classification metrics.

- [`class Distance`](../TFSimilarity/distances/Distance.md): Note: don't forget to add your distance to the DISTANCES list

- [`class Evaluator`](../TFSimilarity/callbacks/Evaluator.md): Evaluates search index performance and calibrates it.

- [`class F1Score`](../TFSimilarity/classification_metrics/F1Score.md): Calculates the harmonic mean of precision and recall.

- [`class FloatTensor`](../TFSimilarity/callbacks/FloatTensor.md): Float tensor 

- [`class Indexer`](../TFSimilarity/indexer/Indexer.md): Indexing system that allows to efficiently find nearest embeddings

- [`class Lookup`](../TFSimilarity/indexer/Lookup.md): Metadata associated with a query match.

- [`class MemoryEvaluator`](../TFSimilarity/callbacks/MemoryEvaluator.md): In memory index performance evaluation and classification.

- [`class MemoryStore`](../TFSimilarity/indexer/MemoryStore.md): Efficient in-memory dataset store

- [`class NMSLibSearch`](../TFSimilarity/indexer/NMSLibSearch.md): Efficiently find nearest embeddings by indexing known embeddings and make

- [`class PandasDataFrame`](../TFSimilarity/indexer/PandasDataFrame.md): Symbolic pandas frame

- [`class Path`](../TFSimilarity/callbacks/Path.md): PurePath subclass that can make system calls.

- [`class RetrievalMetric`](../TFSimilarity/indexer/RetrievalMetric.md): Abstract base class for computing retrieval metrics.

- [`class Search`](../TFSimilarity/indexer/Search.md): Helper class that provides a standard way to create an ABC using

- [`class Store`](../TFSimilarity/indexer/Store.md): Helper class that provides a standard way to create an ABC using

- [`class Tensor`](../TFSimilarity/callbacks/Tensor.md): A `tf.Tensor` represents a multidimensional array of elements.

- [`class defaultdict`](../TFSimilarity/indexer/defaultdict.md): defaultdict(default_factory[, ...]) --> dict with default factory

- [`class deque`](../TFSimilarity/indexer/deque.md): deque([iterable[, maxlen]]) --> deque object

- [`class tqdm`](../TFSimilarity/indexer/tqdm.md): Asynchronous-friendly version of tqdm (Python 3.6+).

## Functions

- [`distance_canonicalizer(...)`](../TFSimilarity/distances/distance_canonicalizer.md): Normalize user requested distance to its matching Distance object.

- [`make_classification_matcher(...)`](../TFSimilarity/indexer/make_classification_matcher.md): Convert classification matcher from str name to object if needed.

- [`make_classification_metric(...)`](../TFSimilarity/callbacks/make_classification_metric.md): Convert classification metric from str name to object if needed.

- [`tabulate(...)`](../TFSimilarity/indexer/tabulate.md): Format a fixed width table for pretty printing.

- [`time(...)`](../TFSimilarity/indexer/time.md): time() -> floating point number

- [`unpack_lookup_distances(...)`](../TFSimilarity/callbacks/unpack_lookup_distances.md)

- [`unpack_lookup_labels(...)`](../TFSimilarity/callbacks/unpack_lookup_labels.md)

