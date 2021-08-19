# TFSimilarity.visualization.viz_neigbors_imgs





Display images nearest neighboors

```python
TFSimilarity.visualization.viz_neigbors_imgs(
    example_class: List[int],
    class_mapping: Dict = None,
    fig_size: Tuple = (24, 4),
    cmap: str = viridis
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>example</b>
</td>
<td>
The data used as query input.
</td>
</tr><tr>
<td>
<b>example_class</b>
</td>
<td>
The class of the data used as query
</td>
</tr><tr>
<td>
<b>neighbors</b>
</td>
<td>
The list of neighbors returned by the lookup()
</td>
</tr><tr>
<td>
<b>class_mapping</b>
</td>
<td>
Dictionary that map the class numerical ids to a class
name. If not set, will display the class numerical id.
Defaults to None.
</td>
</tr><tr>
<td>
<b>fig_size</b>
</td>
<td>
Size of the figure. Defaults to (24, 4).
</td>
</tr><tr>
<td>
<b>cmap</b>
</td>
<td>
Default color scheme for black and white images e.g mnist.
Defaults to 'viridis'.
</td>
</tr>
</table>

