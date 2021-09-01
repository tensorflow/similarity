# TFSimilarity.retrieval_metrics.MapAtK





Mean Average precision (mAP) @K is computed as.

Inherits From: [`RetrievalMetric`](../../TFSimilarity/indexer/RetrievalMetric.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.retrieval_metrics.MapAtK(
    r: Mapping[int, int] = {},
    name: str = map,
    k: int = 1,
    average: str = micro,
    **kwargs
) -> None
```



<!-- Placeholder for "Used in" -->

           k
          ===
          \    rel   . P @j
          /       ij    i
          ===
         j = 1
mAP @k = ------------------
   i           R

Where: K is the number of neighbors in the i_th query result set.
       P is the rolling precision over the i_th query result set.
       R is the cardinality of the target class.
       rel is the relevance mask (indicator function) for the i_th query.
       i represents the i_th query.
       j represents the j_th ranked query result.

AP@K is biased towards the top ranked results and is a function of both the
rank (K) and the class size (R). This metric will be equal to the recall
when all the top results are on contiguous block of TPs.

For example, if the embedding dataset has 100 examples (R) of class 'a',
and our query returns 50 results (K) where the top 10 results are all TPs,
then the AP@50 will be 0.10; however, if instead the bottom 10 ranked
results are all TPs, then the AP@50 will be much lower (0.012) because we
apply a penalty for the 40 FPs that come before the relevant query results.

This metric is useful when we want to ensure that the top ranked results
are relevant to the query.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>r</b>
</td>
<td>
A mapping from class id to the number of examples in the index,
e.g., r[4] = 10 represents 10 indexed examples from class 4
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
Name associated with the metric object, e.g., avg_precision@5
</td>
</tr><tr>
<td>
<b>canonical_name</b>
</td>
<td>
The canonical name associated with metric, e.g.,
avg_precision@K
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
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>r</b>
</td>
<td>
A mapping from class id to the number of examples in the index,
e.g., r[4] = 10 represents 10 indexed examples from class 4
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
Name associated with the metric object, e.g., avg_precision@5
</td>
</tr><tr>
<td>
<b>canonical_name</b>
</td>
<td>
The canonical name associated with metric, e.g.,
avg_precision@K
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
</td>
</tr>
</table>



## Methods

<h3 id="compute">compute</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/map_at_k.py#L98-L151">View source</a>

```python
compute(
    *,
    query_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    match_mask: BoolTensor,
    **kwargs
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
A 1D array of the labels associated with the
embedding queries.
</td>
</tr><tr>
<td>
<b>match_mask</b>
</td>
<td>
A 2D mask where a 1 indicates a match between the
jth query and the kth neighbor and a 0 indicates a mismatch.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Additional compute args
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







