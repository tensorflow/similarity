
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.callbacks.make_metric" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.callbacks.make_metric
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L353-L394">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Covert metric from str name to object if needed.
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.indexer.make_metric`, `TFSimilarity.metrics.make_metric`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.callbacks.make_metric(
    metric: Union[str, <a href="../../TFSimilarity/callbacks/EvalMetric.md"><code>TFSimilarity.callbacks.EvalMetric</code></a>]
) -> <a href="../../TFSimilarity/callbacks/EvalMetric.md"><code>TFSimilarity.callbacks.EvalMetric</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
metric (Union[str, EvalMetric]): EvalMetric() or metric name.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr>
<td>
`ValueError`
</td>
<td>
metric name is invalid.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr>
<td>
`EvalMetric`
</td>
<td>
Instanciated metric if needed.
</td>
</tr>
</table>
