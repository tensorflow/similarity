# TFSimilarity.callbacks.make_metric





Covert metric from str name to object if needed.


```python
TFSimilarity.callbacks.make_metric(
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
metric (Union[str, EvalMetric]): EvalMetric() or metric name.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<b>ValueError</b>
</td>
<td>
metric name is invalid.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
<b>EvalMetric</b>
</td>
<td>
Instanciated metric if needed.
</td>
</tr>
</table>

