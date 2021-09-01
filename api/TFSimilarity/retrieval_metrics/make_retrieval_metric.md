# TFSimilarity.retrieval_metrics.make_retrieval_metric





Convert metric from str name to object if needed.

```python
TFSimilarity.retrieval_metrics.make_retrieval_metric(
    k: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    r: Optional[Mapping[int, int]] = None
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
RetrievalMetric() or metric name.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
The number of nearest neighbors over which the metric is computed.
</td>
</tr><tr>
<td>
<b>distance_threshold</b>
</td>
<td>
The max distance below which a nearest neighbor is
considered a valid match.
</td>
</tr><tr>
<td>
<b>r</b>
</td>
<td>
A mapping from class id to the number of examples in the index,
e.g., r[4] = 10 represents 10 indexed examples from class 4. Only
required for the MAP metric.
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
<b>RetrievalMetric</b>
</td>
<td>
Instantiated metric if needed.
</td>
</tr>
</table>

