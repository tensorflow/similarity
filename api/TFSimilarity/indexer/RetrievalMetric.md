# TFSimilarity.indexer.RetrievalMetric





Abstract base class for computing retrieval metrics.

Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)


```python
TFSimilarity.indexer.RetrievalMetric(
    name: str = ,
    canonical_name: str = ,
    k: int = 1,
    distance_threshold: float = math.inf,
    average: str = micro
) -> None
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
Name associated with the metric object, e.g., recall@5
</td>
</tr><tr>
<td>
<b>canonical_name</b>
</td>
<td>
The canonical name associated with metric,
e.g., recall@K
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
<b>average</b>
</td>
<td>
<i>'micro'</i> Determines the type of averaging performed over the
queries.

    'micro': Calculates metrics globally over all queries.

    'macro': Calculates metrics for each label and takes the unweighted
             mean.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
Name associated with the metric object, e.g., recall@5
</td>
</tr><tr>
<td>
<b>canonical_name</b>
</td>
<td>
The canonical name associated with metric,
e.g., recall@K
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
<b>average</b>
</td>
<td>
<i>'micro'</i> Determines the type of averaging performed over the
queries.

    'micro': Calculates metrics globally over all queries.

    'macro': Calculates metrics for each label and takes the unweighted
             mean.
</td>
</tr>
</table>



## Methods

<h3 id="compute">compute</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/retrieval_metric.py#L76-L99">View source</a>

```python
compute(
    *,
    query_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    lookup_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    lookup_distances: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    match_mask: BoolTensor
) -> <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>
```


Compute the metric


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>query_labels</b>
</td>
<td>
A 1D array of the labels associated with the queries.
</td>
</tr><tr>
<td>
<b>lookup_labels</b>
</td>
<td>
A 2D array where the jth row is the labels
associated with the set of k neighbors for the jth query.
</td>
</tr><tr>
<td>
<b>lookup_distances</b>
</td>
<td>
A 2D array where the jth row is the distances
between the jth query and the set of k neighbors.
</td>
</tr><tr>
<td>
<b>match_mask</b>
</td>
<td>
A 2D mask where a 1 indicates a match between the
jth query and the kth neighbor and a 0 indicates a mismatch.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
metric results.
</td>
</tr>

</table>



<h3 id="get_config">get_config</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/retrieval_metric.py#L68-L74">View source</a>

```python
get_config()
```







