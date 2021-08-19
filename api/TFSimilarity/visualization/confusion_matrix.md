
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.visualization.confusion_matrix" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.visualization.confusion_matrix
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/visualization/confusion_matrix.py#L8-L63">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Plot confusion matrix
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.visualization.confusion_matrix(
    y_pred: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    y_true: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    normalize: bool = (True),
    labels: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a> = None,
    title: str = &#x27;Confusion matrix&#x27;,
    cmap: str = &#x27;Blues&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`y_pred`
</td>
<td>
Model prediction returned by `model.match()`
</td>
</tr><tr>
<td>
`y_true`
</td>
<td>
Expected class_id.
</td>
</tr><tr>
<td>
`normalize`
</td>
<td>
Normalizes matrix values between 0 and 1.
Defaults to True.
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
List of class string label to display instead of the class
numerical ids. Defaults to None.
</td>
</tr><tr>
<td>
`title`
</td>
<td>
Title of the confusion matrix. Defaults to 'Confusion matrix'.
</td>
</tr><tr>
<td>
`cmap`
</td>
<td>
Color schema as CMAP. Defaults to 'Blues'.
</td>
</tr>
</table>
