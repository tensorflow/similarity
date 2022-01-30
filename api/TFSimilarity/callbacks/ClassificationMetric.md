# TFSimilarity.callbacks.ClassificationMetric





Abstract base class for computing classification metrics.

Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)


```python
TFSimilarity.callbacks.ClassificationMetric(
    name: str = ,
    canonical_name: str = ,
    direction: str = max
) -> None
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
Name associated with a specific metric object, e.g.,
accuracy@0.1
</td>
</tr><tr>
<td>
<b>canonical_name</b>
</td>
<td>
The canonical name associated with metric, e.g.,
accuracy
</td>
</tr><tr>
<td>
<b>direction</b>
</td>
<td>
<i>'max','min'</i> the starting point of the search for the
optimal distance threhsold.

* <b>max</b>: Start at the max distance and search decreasing.
* <b>min</b>: Start at the min distance and search increasing.
</td>
</tr>
</table>


<b>ClassificationMetric</b> measure the matching classification between the
query label and the label derived from the set of lookup results.

The <b>compute()</b> method supports computing the metric for a set of values,
where each value represents the counts at a specific distance threshold.

## Methods

<h3 id="compute">compute</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/classification_metrics/classification_metric.py#L65-L91">View source</a>

```python
compute(
    tp: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    fp: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    tn: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    fn: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    count: int
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
```


Compute the classification metric.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>tp</b>
</td>
<td>
A 1D FloatTensor containing the count of True Positives at each
distance threshold.
</td>
</tr><tr>
<td>
<b>fp</b>
</td>
<td>
A 1D FloatTensor containing the count of False Positives at
each distance threshold.
</td>
</tr><tr>
<td>
<b>tn</b>
</td>
<td>
A 1D FloatTensor containing the count of True Negatives at each
distance threshold.
</td>
</tr><tr>
<td>
<b>fn</b>
</td>
<td>
A 1D FloatTensor containing the count of False Negatives at
each distance threshold.
</td>
</tr><tr>
<td>
<b>count</b>
</td>
<td>
The total number of queries
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A 1D FloatTensor containing the metric at each distance threshold.
</td>
</tr>

</table>



<h3 id="get_config">get_config</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/classification_metrics/classification_metric.py#L58-L63">View source</a>

```python
get_config()
```







