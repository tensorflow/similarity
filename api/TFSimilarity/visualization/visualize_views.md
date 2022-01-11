# TFSimilarity.visualization.visualize_views





Display side by side different image views with labels, and predictions

```python
TFSimilarity.visualization.visualize_views(
    num_imgs: int = None,
    views_per_col: int = 4,
    fig_size: Tuple[int, int] = (24, 4),
    max_pixel_value: float = 1.0
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>views</b>
</td>
<td>
Aray of views
</td>
</tr><tr>
<td>
<b>predictions</b>
</td>
<td>
model output.
</td>
</tr><tr>
<td>
<b>labels</b>
</td>
<td>
image labels
</td>
</tr><tr>
<td>
<b>num_imgs</b>
</td>
<td>
number of images to use.
</td>
</tr><tr>
<td>
<b>views_per_col</b>
</td>
<td>
Int, number of images in one row. Defaults to 3.
</td>
</tr><tr>
<td>
<b>max_pixel_value</b>
</td>
<td>
Max expected value for a pixel. Used to scale the image
between [0,1].
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
None.
</td>
</tr>

</table>

