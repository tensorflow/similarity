
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.metrics.make_metrics" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.metrics.make_metrics
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L397-L410">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Convert a list of mixed metrics name and object to a list
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.metrics.make_metrics(
    metrics: List[Union[str, EvalMetric]]
) -> List[<a href="../../TFSimilarity/callbacks/EvalMetric.md"><code>TFSimilarity.callbacks.EvalMetric</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->
of EvalMetrics
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
metrics (List[Union[str, EvalMetric]]): Metric and Metric names
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
List[EvalMetric]: Metrics objects
</td>
</tr>
</table>
