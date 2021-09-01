# TFSimilarity.classification_metrics.NegativePredictiveValue





Calculates the negative predictive value of the query classification.

Inherits From: [`ClassificationMetric`](../../TFSimilarity/callbacks/ClassificationMetric.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.classification_metrics.NegativePredictiveValue(
    name: str = npv
) -> None
```



<!-- Placeholder for "Used in" -->

Computes the NPV given the query classification counts.

$$
FPR = 
rac<i> extrm{true_negatives}}{ extrm{false_negatives</i> +
    extrm<i>true_negatives}</i>
$$

args:
    name: Name associated with a specific metric object, e.g.
    npv@0.1

Usage with <b>tf.similarity.models.SimilarityModel()</b>:

```python
model.calibrate(x=query_examples,
                y=query_labels,
                calibration_metric='fpr')
```

## Methods

<h3 id="compute">compute</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/classification_metrics/negative_predictive_value.py#L47-L74">View source</a>

```python
compute(
    tp: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    fp: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    tn: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    fn: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    count: int
) -> <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
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
A 1D FloatTensor containing the count of False Positives at each
distance threshold.
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
A 1D FloatTensor containing the count of False Negatives at each
distance threshold.
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/classification_metrics/classification_metric.py#L58-L63">View source</a>

```python
get_config()
```







