
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.metrics.batch_class_ratio" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.metrics.batch_class_ratio
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L413-L439">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the average number of examples per class within each batch.
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.metrics.batch_class_ratio(
    sampler: <a href="../../TFSimilarity/metrics/Sampler.md"><code>TFSimilarity.metrics.Sampler</code></a>,
    num_batches: int = 100
) -> float
</code></pre>

<!-- Placeholder for "Used in" -->
Similarity learning requires at least 2 examples per class in each batch.
This is needed in order to construct the triplets. This function
provides the average number of examples per class within each batch and
can be used to check that a sampler is working correctly.
The ratio should be >= 2.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`sampler`
</td>
<td>
A tf.similarity sampler object.
</td>
</tr><tr>
<td>
`num_batches`
</td>
<td>
The number of batches to sample.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The average number of examples per class.
</td>
</tr>
</table>
