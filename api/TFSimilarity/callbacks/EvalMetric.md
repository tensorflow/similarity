# TFSimilarity.callbacks.EvalMetric
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L9-L157">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Helper class that provides a standard way to create an ABC using
Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.indexer.EvalMetric`, `TFSimilarity.metrics.EvalMetric`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.callbacks.EvalMetric(
    direction: str,
    name: str,
    canonical_name: str,
    k: int = None,
    distance_threshold: float = None
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->
inheritance.
## Methods
<h3 id="compute"><code>compute</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/metrics.py#L47-L74">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute(
    max_k: int,
    targets_labels: List[int],
    index_size: int,
    match_ranks: List[int],
    match_distances: List[float],
    lookups: List[List[Lookup]]
) -> Union[int, float]
</code></pre>
Compute the metric

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`max_k`
</td>
<td>
Number of neigboors considers during the retrieval.
</td>
</tr><tr>
<td>
`targets_labels`
</td>
<td>
Expected labels for the query. One label per query.
</td>
</tr><tr>
<td>
`index_size`
</td>
<td>
Total size of the index.
</td>
</tr><tr>
<td>
`match_ranks`
</td>
<td>
Minimal rank at which the targeted label is matched.
For example if the label is matched by the 2nd closest neigboors
then match rank is 2. If there is no match then value 0.
</td>
</tr><tr>
<td>
`match_distances`
</td>
<td>
Minimal distance at which the targeted label is
matched. Mirror the *match_rank* arg.
</td>
</tr><tr>
<td>
`lookups`
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
Filter match ranks to only keep matches between `min_rank`
and `max_rank` below a give distance.
#### Notes:
Neigboors are order by ascending distance.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`match_ranks`
</td>
<td>
Min rank at which the embedding match the correct
label. For example, rank 1 is the closest neigboor,
Rank 5 is the 5th neigboor. There might be neigboors with
higher ranks that are from a different class.
</td>
</tr><tr>
<td>
`min_rank`
</td>
<td>
Minimal rank to keep (inclusive). Defaults to 1.
</td>
</tr><tr>
<td>
`max_rank`
</td>
<td>
Max rank to keep (inclusive). Defaults to None.
If None will keep all ranks above `min_rank`
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



