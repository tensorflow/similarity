# TFSimilarity.callbacks.SplitValidationLoss





Creates the validation callbacks.

```python
TFSimilarity.callbacks.SplitValidationLoss(
    query_labels: Sequence[int],
    target_labels: Sequence[int],
    distance: str = cosine,
    metrics: Sequence[Union[str, ClassificationMetric]] = [binary_accuracy, f1score],
    tb_logdir: str = None,
    k: int = 1,
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>queries</b>
</td>
<td>
Test examples that will be tested against the built index.
</td>
</tr><tr>
<td>
<b>query_labels</b>
</td>
<td>
Queries nearest neighbors expected labels.
</td>
</tr><tr>
<td>
<b>targets</b>
</td>
<td>
Examples that are indexed.
</td>
</tr><tr>
<td>
<b>target_labels</b>
</td>
<td>
Target examples labels.
</td>
</tr><tr>
<td>
<b>known_classes</b>
</td>
<td>
The set of classes seen during training.
</td>
</tr><tr>
<td>
<b>distance</b>
</td>
<td>
Distance function used to compute pairwise distance
between examples embeddings.
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
List of
'tf.similarity.classification_metrics.ClassificationMetric()` to
compute during the evaluation. Defaults to ['binary_accuracy',
'f1score'].
</td>
</tr><tr>
<td>
<b>tb_logdir</b>
</td>
<td>
Where to write TensorBoard logs. Defaults to None.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
The number of nearest neighbors to return for each query.
The lookups are consumed by the Matching Strategy and used to
derive the matching label and distance.
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
<i>'match_nearest', 'match_majority_vote'</i> or
ClassificationMatch object. Defines the classification matching,
e.g., match_nearest will count a True Positive if the query_label
is equal to the label of the nearest neighbor and the distance is
less than or equal to the distance threshold.
</td>
</tr><tr>
<td>
<b>distance_thresholds</b>
</td>
<td>
A 1D tensor denoting the distances points at
which we compute the metrics. If None, distance_thresholds is set
to tf.constant([math.inf])
</td>
</tr>
</table>

