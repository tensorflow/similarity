# TFSimilarity.distance_metrics.masked_max





Computes the maximum values over masked pairwise distances.

```python
TFSimilarity.distance_metrics.masked_max(
    mask: BoolTensor,
    dim: int = 1
```



<!-- Placeholder for "Used in" -->

We need to use this formula to make sure all values are >=0.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>distances</b>
</td>
<td>
2-D float <b>Tensor</b> of [n, n] pairwise distances
</td>
</tr><tr>
<td>
<b>mask</b>
</td>
<td>
2-D Boolean <b>Tensor</b> of [n, n] valid distance size.
</td>
</tr><tr>
<td>
<b>dim</b>
</td>
<td>
The dimension over which to compute the maximum. Defaults to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tuple of Tensors containing the maximum distance value and the arg_max
for each example.
</td>
</tr>

</table>

