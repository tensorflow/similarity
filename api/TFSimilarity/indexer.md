# Module: TFSimilarity.indexer





Index embedding to allow distance based lookup



## Classes

- [`class Distance`](../TFSimilarity/distances/Distance.md): Note: don't forget to add your distance to the DISTANCES list

- [`class EvalMetric`](../TFSimilarity/callbacks/EvalMetric.md): Helper class that provides a standard way to create an ABC using

- [`class Evaluator`](../TFSimilarity/evaluators/Evaluator.md): Evaluates index performance and calibrates it.

- [`class F1Score`](../TFSimilarity/indexer/F1Score.md): Compute the F1 score, also known as balanced F-score or F-measure at K

- [`class FloatTensor`](../TFSimilarity/callbacks/FloatTensor.md): Float tensor 

- [`class Indexer`](../TFSimilarity/indexer/Indexer.md): Indexing system that allows to efficiently find nearest embeddings

- [`class Lookup`](../TFSimilarity/indexer/Lookup.md): Metadata associated with a query match.

- [`class Matcher`](../TFSimilarity/indexer/Matcher.md): Helper class that provides a standard way to create an ABC using

- [`class MemoryEvaluator`](../TFSimilarity/callbacks/MemoryEvaluator.md): In memory index performance evaluation and calibration.

- [`class MemoryTable`](../TFSimilarity/indexer/MemoryTable.md): Efficient in-memory dataset table

- [`class NMSLibMatcher`](../TFSimilarity/indexer/NMSLibMatcher.md): Efficiently find nearest embeddings by indexing known embeddings and make

- [`class PandasDataFrame`](../TFSimilarity/indexer/PandasDataFrame.md): Symbolic pandas frame

- [`class Path`](../TFSimilarity/callbacks/Path.md): PurePath subclass that can make system calls.

- [`class Table`](../TFSimilarity/indexer/Table.md): Helper class that provides a standard way to create an ABC using

- [`class Tensor`](../TFSimilarity/callbacks/Tensor.md): The base class of all dense Tensor objects.

- [`class defaultdict`](../TFSimilarity/indexer/defaultdict.md): defaultdict(default_factory[, ...]) --> dict with default factory

- [`class deque`](../TFSimilarity/indexer/deque.md): deque([iterable[, maxlen]]) --> deque object

- [`class tqdm`](../TFSimilarity/indexer/tqdm.md): Asynchronous-friendly version of tqdm (Python 3.6+).

## Functions

- [`distance_canonicalizer(...)`](../TFSimilarity/distance_metrics/distance_canonicalizer.md): Normalize user requested distance to its matching Distance object.

- [`make_metric(...)`](../TFSimilarity/callbacks/make_metric.md): Covert metric from str name to object if needed.

- [`tabulate(...)`](../TFSimilarity/indexer/tabulate.md): Format a fixed width table for pretty printing.

- [`time(...)`](../TFSimilarity/indexer/time.md): time() -> floating point number

