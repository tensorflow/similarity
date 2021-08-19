
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.visualization.viz_neigbors_imgs" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.visualization.viz_neigbors_imgs
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/visualization/neighbors_viz.py#L6-L53">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Display images nearest neighboors
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.visualization.viz_neigbors_imgs(
    example: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    example_class: List[int],
    neighbors: List[<a href="../../TFSimilarity/indexer/Lookup.md"><code>TFSimilarity.indexer.Lookup</code></a>],
    class_mapping: Dict = None,
    fig_size: Tuple = (24, 4),
    cmap: str = &#x27;viridis&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`example`
</td>
<td>
The data used as query input.
</td>
</tr><tr>
<td>
`example_class`
</td>
<td>
The class of the data used as query
</td>
</tr><tr>
<td>
`neighbors`
</td>
<td>
The list of neighbors returned by the lookup()
</td>
</tr><tr>
<td>
`class_mapping`
</td>
<td>
Dictionary that map the class numerical ids to a class
name. If not set, will display the class numerical id.
Defaults to None.
</td>
</tr><tr>
<td>
`fig_size`
</td>
<td>
Size of the figure. Defaults to (24, 4).
</td>
</tr><tr>
<td>
`cmap`
</td>
<td>
Default color scheme for black and white images e.g mnist.
Defaults to 'viridis'.
</td>
</tr>
</table>
