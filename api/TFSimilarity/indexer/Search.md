# TFSimilarity.indexer.Search





Helper class that provides a standard way to create an ABC using

Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)


```python
TFSimilarity.indexer.Search(
    dim: int,
    verbose: bool,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->
inheritance.

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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/search.py#L42-L58">View source</a>

```python
add(
    embedding: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    idx: int,
    verbose: int = 1,
    **kwargs
)
```


Add a single embedding to the search index.


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
</tr>
</table>



<h3 id="batch_add">batch_add</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/search.py#L60-L76">View source</a>

```python
batch_add(
    embeddings: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    idxs: Sequence[int],
    verbose: int = 1,
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
</tr>
</table>



<h3 id="batch_lookup">batch_lookup</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/search.py#L89-L98">View source</a>

```python
batch_lookup(
    embeddings: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/search.py#L108-L114">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/search.py#L78-L87">View source</a>

```python
lookup(
    embedding: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/search/search.py#L100-L106">View source</a>

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





