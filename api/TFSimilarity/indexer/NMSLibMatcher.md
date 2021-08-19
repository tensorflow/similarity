
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.indexer.NMSLibMatcher" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="batch_add"/>
<meta itemprop="property" content="batch_lookup"/>
<meta itemprop="property" content="load"/>
<meta itemprop="property" content="lookup"/>
<meta itemprop="property" content="save"/>
</div>
# TFSimilarity.indexer.NMSLibMatcher
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L11-L161">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Efficiently find nearest embeddings by indexing known embeddings and make
Inherits From: [`Matcher`](../../TFSimilarity/indexer/Matcher.md), [`ABC`](../../TFSimilarity/distances/ABC.md)
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.matchers.NMSLibMatcher`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.indexer.NMSLibMatcher(
    distance: Union[<a href="../../TFSimilarity/distances/Distance.md"><code>TFSimilarity.distances.Distance</code></a>, str],
    dims: int,
    algorithm: str = &#x27;nmslib_hnsw&#x27;,
    **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->
them searchable using the  [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search library [NMSLIB](https://github.com/nmslib/nmslib).
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`distance`
</td>
<td>
the distance used to compute the distance between
embeddings.
</td>
</tr><tr>
<td>
`dim`
</td>
<td>
the size of the embeddings.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
be verbose.
</td>
</tr>
</table>

## Methods
<h3 id="add"><code>add</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L45-L71">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    idx: int,
    verbose: int = 1,
    build: bool = (True),
    **kwargs
)
</code></pre>
Add an embedding to the index

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embedding`
</td>
<td>
The embedding to index as computed by
the similarity model.
</td>
</tr><tr>
<td>
`idx`
</td>
<td>
Embedding id as in the index table.
Returned with the embedding to allow to lookup
the data associated with a given embedding.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr><tr>
<td>
`build`
</td>
<td>
Rebuild the index after the addition.
Required to make the embedding searchable.
Set to false to save time between successive addition.
Defaults to True.
</td>
</tr>
</table>

<h3 id="batch_add"><code>batch_add</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L73-L102">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_add(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    idxs: List[int],
    verbose: int = 1,
    build: bool = (True),
    **kwargs
)
</code></pre>
Add a batch of embeddings to the matcher.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embeddings`
</td>
<td>
List of embeddings to add to the index.
idxs (int): Embedding ids as in the index table. Returned with
the embeddings to allow to lookup the data associated
with the returned embeddings.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr><tr>
<td>
`build`
</td>
<td>
Rebuild the index after the addition. Required to
make the embeddings searchable. Set to false to save
time between successive addition. Defaults to True.
</td>
</tr>
</table>

<h3 id="batch_lookup"><code>batch_lookup</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L118-L134">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_lookup(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    k: int = 5
) -> Tuple[List[List[int]], List[List[float]]]
</code></pre>
Find embeddings K nearest neighboors embeddings.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embedding`
</td>
<td>
Batch of query embeddings as predicted by the model.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
Number of nearest neighboors embedding to lookup. Defaults to 5.
</td>
</tr>
</table>

<h3 id="load"><code>load</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L145-L152">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load(
    path: str
)
</code></pre>
load index on disk

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`path`
</td>
<td>
where to store the data
</td>
</tr>
</table>

<h3 id="lookup"><code>lookup</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L104-L116">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lookup(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    k: int = 5
) -> Tuple[List[int], List[float]]
</code></pre>
Find embedding K nearest neighboors embeddings.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embedding`
</td>
<td>
Query embedding as predicted by the model.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
Number of nearest neighboors embedding to lookup. Defaults to 5.
</td>
</tr>
</table>

<h3 id="save"><code>save</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/matchers/nmslib_matcher.py#L136-L143">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    path: str
)
</code></pre>
Serializes the index data on disk

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`path`
</td>
<td>
where to store the data
</td>
</tr>
</table>


