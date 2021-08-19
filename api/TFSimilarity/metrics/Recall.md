# TFSimilarity.metrics.Recall





Computing matcher recall at k for a given distance threshold

Inherits From: [`EvalMetric`](../../TFSimilarity/callbacks/EvalMetric.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.metrics.Recall(
    distance_threshold: float = 0.5,
    k: int = 1,
    name=recall
) -> None
```



<!-- Placeholder for "Used in" -->

Recall formula: num_misses / num_queries

## Methods

<h3 id="compute"><code>compute</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L295-L306">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute(
    max_k: int,
    targets_labels: List[int],
    index_size: int,
    match_ranks: List[int],
    match_distances: List[float],
    lookups: List[List[Lookup]]
) -> float
</code></pre>

Compute the metric


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>max_k</b>
</td>
<td>
Number of neigboors considers during the retrieval.
</td>
</tr><tr>
<td>
<b>targets_labels</b>
</td>
<td>
Expected labels for the query. One label per query.
</td>
</tr><tr>
<td>
<b>index_size</b>
</td>
<td>
Total size of the index.
</td>
</tr><tr>
<td>
<b>match_ranks</b>
</td>
<td>
Minimal rank at which the targeted label is matched.
For example if the label is matched by the 2nd closest neigboors
then match rank is 2. If there is no match then value 0.
</td>
</tr><tr>
<td>
<b>match_distances</b>
</td>
<td>
Minimal distance at which the targeted label is
matched. Mirror the *match_rank* arg.
</td>
</tr><tr>
<td>
<b>lookups</b>
</td>
<td>
Full index lookup results to compute more advance metrics.
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



<h3 id="compute_retrival_metrics"><code>compute_retrival_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L137-L157">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_retrival_metrics(
    targets_labels: List[int],
    lookups: List[List[Lookup]]
) -> Tuple[int, int, int, int]
</code></pre>




<h3 id="filter_ranks"><code>filter_ranks</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L89-L135">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>filter_ranks(
    match_ranks: List[int],
    match_distances: List[float],
    min_rank: int = 1,
    max_rank: int = None,
    distance: float = None
) -> List[int]
</code></pre>

Filter match ranks to only keep matches between <b>min_rank</b>
and <b>max_rank</b> below a give distance.

#### Notes:

Neigboors are order by ascending distance.




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>match_ranks</b>
</td>
<td>
Min rank at which the embedding match the correct
label. For example, rank 1 is the closest neigboor,
Rank 5 is the 5th neigboor. There might be neigboors with
higher ranks that are from a different class.
</td>
</tr><tr>
<td>
<b>min_rank</b>
</td>
<td>
Minimal rank to keep (inclusive). Defaults to 1.
</td>
</tr><tr>
<td>
<b>max_rank</b>
</td>
<td>
Max rank to keep (inclusive). Defaults to None.
If None will keep all ranks above <b>min_rank</b>
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
filtered ranks as a dense array with missing elements
removed. len(filtered_ranks) <= len(match_ranks)
</td>
</tr>

</table>



<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L39-L45">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_config(
    config
)
</code></pre>




<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L30-L37">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>






