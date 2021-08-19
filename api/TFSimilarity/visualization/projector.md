# TFSimilarity.visualization.projector





Visualize the embeddings in 2D or 3D using UMAP projection

```python
TFSimilarity.visualization.projector(
    labels: List[Any] = None,
    class_mapping: Optional[List[int]] = None,
    image_size: int = 64,
    tooltips_info: Optional[Dict[str, List[str]]] = None,
    pt_size: int = 3,
    colorize: bool = (True),
    pastel_factor: float = 0.1,
    plot_size: int = 600,
    active_drag: str = box_zoom,
    densmap: bool = (True)
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>embeddings</b>
</td>
<td>
The embeddings outputed by the model that
are to be visualized
</td>
</tr><tr>
<td>
<b>labels</b>
</td>
<td>
Labels associated with the embeddings. If not supplied treat
each example as its own classes.
</td>
</tr><tr>
<td>
<b>class_mapping</b>
</td>
<td>
Dictionary or list that maps the class numerical ids
to their name.
</td>
</tr><tr>
<td>
<b>images</b>
</td>
<td>
Images to display in tooltip on hover. Usually x_test tensor.
</td>
</tr><tr>
<td>
<b>image_size</b>
</td>
<td>
size of the images displayed in the tool tip.
Defaults to 64.
</td>
</tr><tr>
<td>
<b>pt_size</b>
</td>
<td>
Size of the points displayed on the visualization.
Defaults to 3.
</td>
</tr><tr>
<td>
<b>tooltips_info</b>
</td>
<td>
Dictionary of information to display in the tooltips.
</td>
</tr><tr>
<td>
<b>colorize</b>
</td>
<td>
Colorize the clusters. Defaults to true.
</td>
</tr><tr>
<td>
<b>pastel_factor</b>
</td>
<td>
Modify the color palette to be more pastel.
</td>
</tr><tr>
<td>
<b>densmap</b>
</td>
<td>
Use UMAP dense mapper which provides better density
estimation but is a little slower. Defaults to True.
</td>
</tr>
</table>

