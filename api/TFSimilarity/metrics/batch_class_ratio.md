# TFSimilarity.metrics.batch_class_ratio





Computes the average number of examples per class within each batch.

```python
TFSimilarity.metrics.batch_class_ratio(
```

    num_batches: int = 100
) -> float
```



<!-- Placeholder for "Used in" -->

Similarity learning requires at least 2 examples per class in each batch.
This is needed in order to construct the triplets. This function
provides the average number of examples per class within each batch and
can be used to check that a sampler is working correctly.

The ratio should be >= 2.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>sampler</b>
</td>
<td>
A tf.similarity sampler object.
</td>
</tr><tr>
<td>
<b>num_batches</b>
</td>
<td>
The number of batches to sample.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The average number of examples per class.
</td>
</tr>

</table>

