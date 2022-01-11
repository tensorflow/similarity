# TFSimilarity.indexer.CalibrationResults





Cutpoints and thresholds associated with a calibration.

```python
TFSimilarity.indexer.CalibrationResults(
    cutpoints: Mapping[str, Mapping[str, Union[str, float, int]]],
    thresholds: Mapping[str, Sequence[Union[float, int]]]
)
```



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>cutpoints</b>
</td>
<td>
A Dict mapping the cutpoint name to a Dict containing the
ClassificationMetric values associated with a particular distance
threshold, e.g., 'optimal' : <i>'acc': 0.90, 'f1': 0.92</i>.
</td>
</tr><tr>
<td>
<b>thresholds</b>
</td>
<td>
A Dict mapping ClassificationMetric names to a list
containing the metric's value computed at each of the distance
thresholds, e.g., <i>'f1': [0.99, 0.80], 'distance': [0.0, 1.0]</i>.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__">__eq__</h3>

```python
__eq__(
    other
)
```







