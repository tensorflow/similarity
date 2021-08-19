<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.metrics" />
<meta itemprop="path" content="Stable" />
</div>
# Module: TFSimilarity.metrics
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



## Classes
[`class ABC`](../TFSimilarity/distances/ABC.md): Helper class that provides a standard way to create an ABC using
[`class Accuracy`](../TFSimilarity/metrics/Accuracy.md): How many correct matches are returned for the given set pf parameters
[`class EvalMetric`](../TFSimilarity/callbacks/EvalMetric.md): Helper class that provides a standard way to create an ABC using
[`class F1Score`](../TFSimilarity/indexer/F1Score.md): Compute the F1 score, also known as balanced F-score or F-measure at K
[`class FIXMEPrecision`](../TFSimilarity/metrics/FIXMEPrecision.md): Compute the precision of the matches.
[`class Lookup`](../TFSimilarity/indexer/Lookup.md): Metadata associated with a query match.
[`class MaxRank`](../TFSimilarity/metrics/MaxRank.md): Helper class that provides a standard way to create an ABC using
[`class MeanRank`](../TFSimilarity/metrics/MeanRank.md): Helper class that provides a standard way to create an ABC using
[`class MinRank`](../TFSimilarity/metrics/MinRank.md): Helper class that provides a standard way to create an ABC using
[`class Recall`](../TFSimilarity/metrics/Recall.md): Computing matcher recall at k for a given distance threshold
[`class Sampler`](../TFSimilarity/metrics/Sampler.md): Base object for fitting to a sequence of data, such as a dataset.
## Functions
[`batch_class_ratio(...)`](../TFSimilarity/metrics/batch_class_ratio.md): Computes the average number of examples per class within each batch.
[`make_metric(...)`](../TFSimilarity/callbacks/make_metric.md): Covert metric from str name to object if needed.
[`make_metrics(...)`](../TFSimilarity/metrics/make_metrics.md): Convert a list of mixed metrics name and object to a list
