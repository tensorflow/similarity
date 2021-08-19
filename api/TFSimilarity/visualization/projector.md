
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.visualization.projector" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.visualization.projector
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/visualization/projector.py#L57-L192">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Visualize the embeddings in 2D or 3D using UMAP projection
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.visualization.projector(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    labels: List[Any] = None,
    class_mapping: Optional[List[int]] = None,
    images: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>] = None,
    image_size: int = 64,
    tooltips_info: Optional[Dict[str, List[str]]] = None,
    pt_size: int = 3,
    colorize: bool = (True),
    pastel_factor: float = 0.1,
    plot_size: int = 600,
    active_drag: str = &#x27;box_zoom&#x27;,
    densmap: bool = (True)
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`embeddings`
</td>
<td>
The embeddings outputed by the model that
are to be visualized
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
Labels associated with the embeddings. If not supplied treat
each example as its own classes.
</td>
</tr><tr>
<td>
`class_mapping`
</td>
<td>
Dictionary or list that maps the class numerical ids
to their name.
</td>
</tr><tr>
<td>
`images`
</td>
<td>
Images to display in tooltip on hover. Usually x_test tensor.
</td>
</tr><tr>
<td>
`image_size`
</td>
<td>
size of the images displayed in the tool tip.
Defaults to 64.
</td>
</tr><tr>
<td>
`pt_size`
</td>
<td>
Size of the points displayed on the visualization.
Defaults to 3.
</td>
</tr><tr>
<td>
`tooltips_info`
</td>
<td>
Dictionary of information to display in the tooltips.
</td>
</tr><tr>
<td>
`colorize`
</td>
<td>
Colorize the clusters. Defaults to true.
</td>
</tr><tr>
<td>
`pastel_factor`
</td>
<td>
Modify the color palette to be more pastel.
</td>
</tr><tr>
<td>
`densmap`
</td>
<td>
Use UMAP dense mapper which provides better density
estimation but is a little slower. Defaults to True.
</td>
</tr>
</table>
