# TFSimilarity.distance_metrics.masked_min
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/algebra.py#L34-L56">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the minimal values over masked pairwise distances.
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.distance_metrics.masked_min(
    distances: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    mask: BoolTensor,
    dim: int = 1
) -> Tuple[<a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>, <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`distances`
</td>
<td>
2-D float `Tensor` of [n, n] pairwise distances
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
2-D Boolean `Tensor` of [n, n] valid distance size.
</td>
</tr><tr>
<td>
`dim`
</td>
<td>
The dimension over which to compute the minimum. Defaults to 1.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tuple of Tensors containing the minimal distance value and the arg_min
for each example.
</td>
</tr>
</table>
