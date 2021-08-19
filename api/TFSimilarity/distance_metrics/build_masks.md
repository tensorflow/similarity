# TFSimilarity.distance_metrics.build_masks
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/algebra.py#L59-L82">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Build masks that allows to select only the positive or negatives
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.distance_metrics.build_masks(
    labels: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    batch_size: int
) -> Tuple[BoolTensor, BoolTensor]
</code></pre>

<!-- Placeholder for "Used in" -->
embeddings.
Args:
    labels: 1D int `Tensor` that contains the class ids.
    batch_size: size of the batch.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tuple of Tensors containing the positive_mask and negative_mask
</td>
</tr>
</table>
