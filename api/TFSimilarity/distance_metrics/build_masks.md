# TFSimilarity.distance_metrics.build_masks





Build masks that allows to select only the positive or negatives

```python
TFSimilarity.distance_metrics.build_masks(
```

    batch_size: int
) -> Tuple[BoolTensor, BoolTensor]
```



<!-- Placeholder for "Used in" -->
embeddings.
Args:
    labels: 1D int <b>Tensor</b> that contains the class ids.
    batch_size: size of the batch.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tuple of Tensors containing the positive_mask and negative_mask
</td>
</tr>

</table>

