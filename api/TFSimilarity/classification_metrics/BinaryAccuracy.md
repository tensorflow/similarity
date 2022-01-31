# TFSimilarity.classification_metrics.BinaryAccuracy





Calculates how often the query label matches the derived lookup label.

Inherits From: [`ClassificationMetric`](../../TFSimilarity/callbacks/ClassificationMetric.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.classification_metrics.BinaryAccuracy(
    name: str = binary_accuracy
) -> None
```



<!-- Placeholder for "Used in" -->

Accuracy is technically (TP+TN)/(TP+FP+TN+FN), but here we filter all
queries above the distance threshold. In the case of binary matching, this
makes all the TPs and FPs below the distance threshold and all the TNs and
FNs above the distance threshold.

As we are only concerned with the matches below the distance threshold, the
accuracy simplifies to TP/(TP+FP) and is equivalent to the precision with
respect to the unfiltered queries. However, we also want to consider the
query coverage at the distance threshold, i.e., the percentage of queries
that retrun a match, computed as (TP+FP)/(TP+FP+TN+FN). Therefore, we can
take $ precision    imes query_coverage $ to produce a measure that capture
the precision scaled by the query coverage. This simplifies down to the
binary accuracy presented here, giving TP/(TP+FP+TN+FN).

args:
    name: Name associated with a specific metric object, e.g.,
    binary_accuracy@0.1

Usage with <b>tf.similarity.models.SimilarityModel()</b>:

```python
model.calibrate(x=query_examples,
                y=query_labels,
                calibration_metric='binary_accuracy')
```

## Methods

<h3 id="compute">compute</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/classification_metrics/binary_accuracy.py#L53-L84">View source</a>

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

The <b>compute()</b> method supports computing the metric for a set of
values, where each value represents the counts at a specific distance
threshold.

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







