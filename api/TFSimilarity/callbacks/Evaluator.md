# TFSimilarity.callbacks.Evaluator





Evaluates search index performance and calibrates it.

Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)

<!-- Placeholder for "Used in" -->

Index evaluators are derived from this abstract class to allow users to
override the evaluation to use additional data or interface
with existing evaluation system. For example allowing to fetch data from
a remote database.

## Methods

<h3 id="calibrate">calibrate</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/evaluators/evaluator.py#L108-L153">View source</a>

```python
calibrate(
    target_labels: Sequence[int],
    lookups: Sequence[Sequence[Lookup]],
    thresholds_targets: MutableMapping[str, float],
    calibration_metric: <a href="../../TFSimilarity/callbacks/ClassificationMetric.md">TFSimilarity.callbacks.ClassificationMetric```
</a>,
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>],
    extra_metrics: Sequence[<a href="../../TFSimilarity/callbacks/ClassificationMetric.md">TFSimilarity.callbacks.ClassificationMetric```
</a>] = [],
    distance_rounding: int = 8,
    metric_rounding: int = 6,
    verbose: int = 1
) -> <a href="../../TFSimilarity/indexer/CalibrationResults.md">TFSimilarity.indexer.CalibrationResults```
</a>
```


Computes the distances thresholds that the classification must match to
meet a fixed target.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>target_labels</b>
</td>
<td>
Sequence of expected labels for the lookups.
</td>
</tr><tr>
<td>
<b>lookup</b>
</td>
<td>
Sequence of lookup results as produced by the
<b>Index.batch_lookup()</b> method.
</td>
</tr><tr>
<td>
<b>thresholds_targets</b>
</td>
<td>
classification metrics thresholds that are
targeted. The function will find the closed distance value.
</td>
</tr><tr>
<td>
<b>calibration_metric</b>
</td>
<td>
Classification metric used for calibration.
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
<i>'match_nearest', 'match_majority_vote'</i> or
ClassificationMatch object. Defines the classification matching,
e.g., match_nearest will count a True Positive if the query_label
is equal to the label of the nearest neighbor and the distance is
less than or equal to the distance threshold.
</td>
</tr><tr>
<td>
<b>extra_metrics</b>
</td>
<td>
Additional metrics that should be computed and
reported as part of the classification. Defaults to [].
</td>
</tr><tr>
<td>
<b>distance_rounding</b>
</td>
<td>
How many digit to consider to
decide if the distance changed. Defaults to 8.
</td>
</tr><tr>
<td>
<b>metric_rounding</b>
</td>
<td>
How many digit to consider to decide if
the metric changed. Defaults to 6.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
CalibrationResults containing the thresholds and cutpoints Dicts.
</td>
</tr>

</table>



<h3 id="evaluate_classification">evaluate_classification</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/evaluators/evaluator.py#L62-L106">View source</a>

```python
evaluate_classification(
    query_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    lookup_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    lookup_distances: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    distance_thresholds: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    metrics: Sequence[<a href="../../TFSimilarity/callbacks/ClassificationMetric.md">TFSimilarity.callbacks.ClassificationMetric```
</a>],
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>],
    distance_rounding: int = 8,
    verbose: int = 1
) -> Dict[str, np.ndarray]
```


Evaluate the classification performance.

Compute the classification metrics given a set of queries, lookups, and
distance thresholds.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>query_labels</b>
</td>
<td>
Sequence of expected labels for the lookups.
</td>
</tr><tr>
<td>
<b>lookup_labels</b>
</td>
<td>
A 2D tensor where the jth row is the labels
associated with the set of k neighbors for the jth query.
</td>
</tr><tr>
<td>
<b>lookup_distances</b>
</td>
<td>
A 2D tensor where the jth row is the distances
between the jth query and the set of k neighbors.
</td>
</tr><tr>
<td>
<b>distance_thresholds</b>
</td>
<td>
A 1D tensor denoting the distances points at
which we compute the metrics.
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
The set of classification metrics.
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
<i>'match_nearest', 'match_majority_vote'</i> or
ClassificationMatch object. Defines the classification matching,
e.g., match_nearest will count a True Positive if the query_label
is equal to the label of the nearest neighbor and the distance is
less than or equal to the distance threshold.
</td>
</tr><tr>
<td>
<b>distance_rounding</b>
</td>
<td>
How many digit to consider to
decide if the distance changed. Defaults to 8.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Mapping from metric name to the list of values computed for each
distance threshold.
</td>
</tr>

</table>



<h3 id="evaluate_retrieval">evaluate_retrieval</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/evaluators/evaluator.py#L36-L60">View source</a>

```python
evaluate_retrieval(
    target_labels: Sequence[int],
    lookups: Sequence[Sequence[Lookup]],
    retrieval_metrics: Sequence[<a href="../../TFSimilarity/indexer/RetrievalMetric.md">TFSimilarity.indexer.RetrievalMetric```
</a>],
    distance_rounding: int = 8
) -> Dict[str, np.ndarray]
```


Evaluates lookup performances against a supplied set of metrics


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>target_labels</b>
</td>
<td>
Sequence of the expected labels to match.
</td>
</tr><tr>
<td>
<b>lookups</b>
</td>
<td>
Sequence of lookup results as produced by the
<b>Index().batch_lookup()</b> method.
</td>
</tr><tr>
<td>
<b>retrieval_metrics</b>
</td>
<td>
Sequence of <b>RetrievalMetric()</b> to evaluate
lookup matches against.
</td>
</tr><tr>
<td>
<b>distance_rounding</b>
</td>
<td>
How many digit to consider to decide if
the distance changed. Defaults to 8.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dictionary of metric results where keys are the metric names and
values are the metrics values.
</td>
</tr>

</table>





