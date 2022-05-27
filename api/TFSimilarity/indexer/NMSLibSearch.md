# TFSimilarity.indexer.NMSLibSearch





Efficiently find nearest embeddings by indexing known embeddings and make

Inherits From: [`Search`](../../TFSimilarity/indexer/Search.md), [`ABC`](../../TFSimilarity/distances/ABC.md)


```python
TFSimilarity.indexer.NMSLibSearch(
    dims: int,
    algorithm: str = nmslib_hnsw,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->
them searchable using the  [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search library [NMSLIB](https://github.com/nmslib/nmslib).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>distance</b>
</td>
<td>
the distance used to compute the distance between
embeddings.
</td>
</tr><tr>
<td>
<b>dim</b>
</td>
<td>
the size of the embeddings.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
be verbose.
</td>
</tr>
</table>



## Methods

<h3 id="add">add</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/nmslib_search.py#L63-L91">View source</a>

```python
add(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    idx: int,
    verbose: int = 1,
    build: bool = True,
    **kwargs
)
```


Add an embedding to the index


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embedding</b>
</td>
<td>
The embedding to index as computed by
the similarity model.
</td>
</tr><tr>
<td>
<b>idx</b>
</td>
<td>
Embedding id as in the index table.
Returned with the embedding to allow to lookup
the data associated with a given embedding.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr><tr>
<td>
<b>build</b>
</td>
<td>
Rebuild the index after the addition.
Required to make the embedding searchable.
Set to false to save time between successive addition.
Defaults to True.
</td>
</tr>
</table>



<h3 id="batch_add">batch_add</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/nmslib_search.py#L93-L124">View source</a>

```python
batch_add(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    idxs: Sequence[int],
    verbose: int = 1,
    build: bool = True,
    **kwargs
)
```


Add a batch of embeddings to the search index.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embeddings</b>
</td>
<td>
List of embeddings to add to the index.

idxs (int): Embedding ids as in the index table. Returned with
the embeddings to allow to lookup the data associated
with the returned embeddings.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr><tr>
<td>
<b>build</b>
</td>
<td>
Rebuild the index after the addition. Required to
make the embeddings searchable. Set to false to save
time between successive addition. Defaults to True.
</td>
</tr>
</table>



<h3 id="batch_lookup">batch_lookup</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/nmslib_search.py#L140-L156">View source</a>

```python
batch_lookup(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    k: int = 5
) -> Tuple[List[List[int]], List[List[float]]]
```


Find embeddings K nearest neighboors embeddings.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embedding</b>
</td>
<td>
Batch of query embeddings as predicted by the model.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
Number of nearest neighboors embedding to lookup. Defaults to 5.
</td>
</tr>
</table>



<h3 id="load">load</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/nmslib_search.py#L182-L205">View source</a>

```python
load(
    path: str
)
```


load index on disk


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>path</b>
</td>
<td>
where to store the data
</td>
</tr>
</table>



<h3 id="lookup">lookup</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/nmslib_search.py#L126-L138">View source</a>

```python
lookup(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    k: int = 5
) -> Tuple[List[int], List[float]]
```


Find embedding K nearest neighboors embeddings.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embedding</b>
</td>
<td>
Query embedding as predicted by the model.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
Number of nearest neighboors embedding to lookup. Defaults to 5.
</td>
</tr>
</table>



<h3 id="save">save</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/nmslib_search.py#L158-L180">View source</a>

```python
save(
    path: str
)
```


Serializes the index data on disk


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>path</b>
</td>
<td>
where to store the data
</td>
</tr>
</table>





