# TFSimilarity.retrieval_metrics.BNDCG





Binary normalized discounted cumulative gain.

Inherits From: [`RetrievalMetric`](../../TFSimilarity/indexer/RetrievalMetric.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.retrieval_metrics.BNDCG(
    name: str = ndcg,
    k: int = 5,
    distance_threshold: float = math.inf,
    **kwargs
) -> None
```



<!-- Placeholder for "Used in" -->

This is normalized discounted cumulative gain where the relevancy weights
are binary, i.e., either a correct match or an incorrect match.

The NDCG is a score between [0,1] representing the rank weighted results.
The DCG represents the sum of the correct matches weighted by the log2 of
the rank and is normalized by the 'ideal DCG'. The IDCG is computed as the
match_mask, sorted descending, weighted by the log2 of the post sorting rank
order. This metric takes into account both the correctness of the match and
the position.

The normalized DCG is computed as:

$$
nDCG_<i>p} = \frac{DCG_{p}}{IDCG_{p}</i>
$$

The DCG is computed for each query using the match_mask as:

$$
DCG_<i>p} = \sum_{i=1}^{p} \frac{match_mask_{i}}{\log_{2}(i+1)</i>
$$

The IDCG uses the same equation but sorts the match_mask descending
along axis=-1.

Additionally, all positive matches with a distance above the threshold are
set to 0, and the closest K matches are taken.

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

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/bndcg.py#L87-L160">View source</a>

```python
compute(
    *,
    query_labels: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    lookup_distances: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    match_mask: <a href="../../TFSimilarity/utils/BoolTensor.md">TFSimilarity.utils.BoolTensor```
</a>,
    **kwargs
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
```


Compute the metric

Computes the binary NDCG. The query labels are only used when the
averaging is set to "macro".

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
A rank 0 tensor containing the metric.
</td>
</tr>

</table>



<h3 id="get_config">get_config</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/retrieval_metrics/retrieval_metric.py#L79-L85">View source</a>

```python
get_config()
```







