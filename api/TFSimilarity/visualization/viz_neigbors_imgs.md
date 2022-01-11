# TFSimilarity.visualization.viz_neigbors_imgs





Display images nearest neighboors

```python
TFSimilarity.visualization.viz_neigbors_imgs(
    example_class: int,
    class_mapping: Optional[Mapping[int, str]] = None,
    fig_size: Tuple[int, int] = (24, 4),
    cmap: str = viridis,
    show: bool = True
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
Mapping from class numerical ids to a class name. If not
set, the plot will display the class numerical id instead.
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
</tr><tr>
<td>
<b>show</b>
</td>
<td>
If the plot is going to be shown or not. Defaults to True.
</td>
</tr>
</table>

