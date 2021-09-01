# Module: TFSimilarity.classification_metrics





Classification metrics measure matching classification quality between a

set query examples and a set of indexed examples. 

## Classes

- [`class BinaryAccuracy`](../TFSimilarity/classification_metrics/BinaryAccuracy.md): Calculates how often the query label matches the derived lookup label.

- [`class ClassificationMetric`](../TFSimilarity/callbacks/ClassificationMetric.md): Abstract base class for computing classification metrics.

- [`class F1Score`](../TFSimilarity/classification_metrics/F1Score.md): Calculates the harmonic mean of precision and recall.

- [`class FalsePositiveRate`](../TFSimilarity/classification_metrics/FalsePositiveRate.md): Calculates the false positive rate of the query classification.

- [`class NegativePredictiveValue`](../TFSimilarity/classification_metrics/NegativePredictiveValue.md): Calculates the negative predictive value of the query classification.

- [`class Precision`](../TFSimilarity/classification_metrics/Precision.md): Calculates the precision of the query classification.

- [`class Recall`](../TFSimilarity/classification_metrics/Recall.md): Calculates the recall of the query classification.

## Functions

- [`make_classification_metric(...)`](../TFSimilarity/callbacks/make_classification_metric.md): Convert classification metric from str name to object if needed.

