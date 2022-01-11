# TFSimilarity.samplers.select_examples





Randomly select at most N examples per class

```python
TFSimilarity.samplers.select_examples(
    class_list: Sequence[int] = None,
    num_examples_per_class: int = None
) -> Tuple[np.ndarray, np.ndarray]
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A 2-D Tensor containing the data.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A 1-D Tensor containing the labels.
</td>
</tr><tr>
<td>
<b>class_list</b>
</td>
<td>
Filter the list of examples to only keep thoses those who
belong to the supplied class list. In no class is supplied, keep
examples for all the classes. Default to None - keep all the examples.
</td>
</tr><tr>
<td>
<b>num_examples_per_class</b>
</td>
<td>
Restrict the number of examples for EACH
class to num_examples_per_class if set. If not set, all the available
examples are selected. Defaults to None - no selection.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tuple containing the subset of x and y.
</td>
</tr>

</table>

