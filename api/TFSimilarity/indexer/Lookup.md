# TFSimilarity.indexer.Lookup
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/types.py#L115-L150">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Metadata associated with a query match.
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.metrics.Lookup`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.indexer.Lookup(
    rank: int,
    distance: float,
    label: Optional[int] = None,
    embedding: Optional[np.ndarray] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>] = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`rank`
</td>
<td>
Rank of the match with respect to the query distance.
</td>
</tr><tr>
<td>
`distance`
</td>
<td>
The distance from the match to the query.
</td>
</tr><tr>
<td>
`label`
</td>
<td>
The label associated with the match. Default None.
</td>
</tr><tr>
<td>
`embedding`
</td>
<td>
The embedded match vector. Default None.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
The original Tensor representation of the match result.
Default None.
</td>
</tr>
</table>


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`rank`
</td>
<td>
Rank of the match with respect to the query distance.
</td>
</tr><tr>
<td>
`distance`
</td>
<td>
The distance from the match to the query.
</td>
</tr><tr>
<td>
`label`
</td>
<td>
The label associated with the match. Default None.
</td>
</tr><tr>
<td>
`embedding`
</td>
<td>
The embedded match vector. Default None.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
The original Tensor representation of the match result.
Default None.
</td>
</tr>
</table>

## Methods
<h3 id="__eq__"><code>__eq__</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/types.py#L136-L150">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>
Return self==value.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>
<tr>
<td>
data<a id="data"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
embedding<a id="embedding"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
label<a id="label"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
