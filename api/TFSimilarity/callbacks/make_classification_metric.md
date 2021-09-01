# TFSimilarity.callbacks.make_classification_metric





Convert classification metric from str name to object if needed.


```python
TFSimilarity.callbacks.make_classification_metric(
    name: str = 
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>metric</b>
</td>
<td>
ClassificationMetric() or metric name.
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
Unknown metric name: <i>metric</i>, typo?
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
<b>ClassificationMetric</b>
</td>
<td>
Instantiated metric if needed.
</td>
</tr>
</table>

