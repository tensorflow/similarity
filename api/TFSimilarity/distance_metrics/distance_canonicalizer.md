# TFSimilarity.distance_metrics.distance_canonicalizer
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L212-L247">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Normalize user requested distance to its matching Distance object.
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.distances.distance_canonicalizer`, `TFSimilarity.indexer.distance_canonicalizer`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.distance_metrics.distance_canonicalizer(
    user_distance: Union[<a href="../../TFSimilarity/distances/Distance.md"><code>TFSimilarity.distances.Distance</code></a>, str]
) -> <a href="../../TFSimilarity/distances/Distance.md"><code>TFSimilarity.distances.Distance</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`user_distance`
</td>
<td>
Requested distance either by name or by object
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr>
<td>
`Distance`
</td>
<td>
Requested object name.
</td>
</tr>
</table>
