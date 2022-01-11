# TFSimilarity.retrieval_metrics.PrecisionAtK





Precision@K is computed as.

Inherits From: [`RetrievalMetric`](../../TFSimilarity/indexer/RetrievalMetric.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.retrieval_metrics.PrecisionAtK(
    name: str = precision,
    k: int = 5,
    **kwargs
) -> None
```



<!-- Placeholder for "Used in" -->

$$
P_i@k = 
rac<i>TP_i}{TP_i+FP_i</i> = 
rac<i>\sum_{j = 1}^{k} {rel_i_j}}{K</i>
$$

Where: K is the number of neighbors in the i_th query result set.
       rel is the relevance mask (indicator function) for the i_th query.
       i represents the i_th query.
       j represents the j_th ranked query result.

P@K is unordered and does not take into account the rank of the TP results.

This metric is useful when we are interested in evaluating the embedding
within the context of a kNN classifier or as part of a clustering method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
Name associated with the metric object, e.g., precision@5
</td>
</tr><tr>
<td>
<b>canonical_name</b>
</td>
<td>
The canonical name associated with metric,
e.g., precision@K
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
<i>'micro', 'macro'</i> Determines the type of averaging performed
on the data.

* 'micro': Calculates metrics globally over all data.

* 'macro': Calculates metrics for each label and takes the unweighted
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

</td>
</tr>
</table>



## Methods

<h3 id="compute">compute</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/precision_at_k.py#L64-L106">View source</a>

```python
compute(
    *,
    query_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    match_mask: <a href="../../TFSimilarity/utils/BoolTensor.md">TFSimilarity.utils.BoolTensor```
</a>,
    **kwargs
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
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
Additional compute args.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A rank 0 tensor containing the metric.
</td>
</tr>

</table>



<h3 id="get_config">get_config</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/retrieval_metric.py#L79-L85">View source</a>

```python
get_config()
```







