
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.evaluators.Evaluator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="calibrate"/>
<meta itemprop="property" content="evaluate"/>
</div>
# TFSimilarity.evaluators.Evaluator
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/evaluators/evaluator.py#L7-L81">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Evaluates index performance and calibrates it.
Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.indexer.Evaluator`</p>
</p>
</section>
<!-- Placeholder for "Used in" -->
Index evaluators are derived from this abstract class to allow users to
override the evaluation module to use additional data or interface
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
`index_size`
</td>
<td>
Index size.
</td>
</tr><tr>
<td>
`calibration_metric`
</td>
<td>
Metric used for calibration.
</td>
</tr><tr>
<td>
`thresholds_targets`
</td>
<td>
Calibration metrics thresholds that are
targeted. The function will find the closed distance value.
</td>
</tr><tr>
<td>
`targets_labels`
</td>
<td>
List of expected labels for the lookups.
</td>
</tr><tr>
<td>
`lookup`
</td>
<td>
List of lookup results as produced by the
`Index.batch_lookup()` method.
</td>
</tr><tr>
<td>
`extra_metrics`
</td>
<td>
Additional metrics that should be computed and
reported as part of the calibration. Defaults to [].
</td>
</tr><tr>
<td>
`distance_rounding`
</td>
<td>
How many digit to consider to
decide if the distance changed. Defaults to 8.
</td>
</tr><tr>
<td>
`metric_rounding`
</td>
<td>
How many digit to consider to decide if
the metric changed. Defaults to 6.
</td>
</tr><tr>
<td>
`verbose`
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
`index_size`
</td>
<td>
Size of the search index.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
List of `EvalMetric()` to evaluate lookup matches against.
</td>
</tr><tr>
<td>
`targets_labels`
</td>
<td>
List of the expected labels to match.
</td>
</tr><tr>
<td>
`lookups`
</td>
<td>
List of lookup results as produced by the
`Index().batch_lookup()` method.
</td>
</tr><tr>
<td>
`distance_rounding`
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


