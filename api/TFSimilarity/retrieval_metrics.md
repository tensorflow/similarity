# Module: TFSimilarity.retrieval_metrics





Retrieval metrics measure the quality of the embedding space given a

set query examples and a set of indexed examples. Informally it can be thought
of as how well the space is clustered among other things.

## Classes

- [`class BNDCG`](../TFSimilarity/retrieval_metrics/BNDCG.md): Binary normalized discounted cumulative gain.

- [`class MapAtK`](../TFSimilarity/retrieval_metrics/MapAtK.md): Mean Average precision (mAP) @K is computed as.

- [`class PrecisionAtK`](../TFSimilarity/retrieval_metrics/PrecisionAtK.md): Precision@K is computed as.

- [`class RecallAtK`](../TFSimilarity/retrieval_metrics/RecallAtK.md): The metric learning version of Recall@K.

- [`class RetrievalMetric`](../TFSimilarity/indexer/RetrievalMetric.md): Abstract base class for computing retrieval metrics.

## Functions

- [`make_retrieval_metric(...)`](../TFSimilarity/retrieval_metrics/make_retrieval_metric.md): Convert metric from str name to object if needed.

