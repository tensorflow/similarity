# TFSimilarity.evaluators.Evaluator





Evaluates index performance and calibrates it.

Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)

<!-- Placeholder for "Used in" -->

Index evaluators are derived from this abstract class to allow users to
override the evaluation to use additional data or interface
with existing evaluation system. For example allowing to fetch data from
a remote database.

## Methods

<h3 id="calibrate"><code>calibrate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/evaluators/evaluator.py#L44-L81">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>calibrate(
    index_size: int,
    calibration_metric: <a href="../../TFSimilarity/callbacks/EvalMetric.md"><code>TFSimilarity.callbacks.EvalMetric</code></a>,
    thresholds_targets: Dict[str, float],
    targets_labels: List[int],
    lookups: List[List[Lookup]],
    extra_metrics: List[Union[str, EvalMetric]] = [],
    distance_rounding: int = 8,
    metric_rounding: int = 6,
    verbose: int = 1
)
</code></pre>

Computes the distances thresholds that the calibration much match to
meet fixed target.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>index_size</b>
</td>
<td>
Index size.
</td>
</tr><tr>
<td>
<b>calibration_metric</b>
</td>
<td>
Metric used for calibration.
</td>
</tr><tr>
<td>
<b>thresholds_targets</b>
</td>
<td>
Calibration metrics thresholds that are
targeted. The function will find the closed distance value.
</td>
</tr><tr>
<td>
<b>targets_labels</b>
</td>
<td>
List of expected labels for the lookups.
</td>
</tr><tr>
<td>
<b>lookup</b>
</td>
<td>
List of lookup results as produced by the
<b>Index.batch_lookup()</b> method.
</td>
</tr><tr>
<td>
<b>extra_metrics</b>
</td>
<td>
Additional metrics that should be computed and
reported as part of the calibration. Defaults to [].
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



<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/evaluators/evaluator.py#L16-L42">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    index_size: int,
    metrics: List[Union[str, EvalMetric]],
    targets_labels: List[int],
    lookups: List[List[Lookup]],
    distance_rounding: int = 8
) -> Dict[str, Union[float, int]]
</code></pre>

Evaluates lookup performances against a supplied set of metrics


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>index_size</b>
</td>
<td>
Size of the search index.
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
List of <b>EvalMetric()</b> to evaluate lookup matches against.
</td>
</tr><tr>
<td>
<b>targets_labels</b>
</td>
<td>
List of the expected labels to match.
</td>
</tr><tr>
<td>
<b>lookups</b>
</td>
<td>
List of lookup results as produced by the
<b>Index().batch_lookup()</b> method.
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
Dictionnary of metric results where keys are the metric
names and values are the metrics values.
</td>
</tr>

</table>





