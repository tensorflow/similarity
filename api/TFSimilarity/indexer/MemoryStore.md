# TFSimilarity.indexer.MemoryStore





Efficient in-memory dataset store

Inherits From: [`Store`](../../TFSimilarity/indexer/Store.md), [`ABC`](../../TFSimilarity/distances/ABC.md)


```python
TFSimilarity.indexer.MemoryStore() -> None
```



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="add">add</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L39-L62">View source</a>

```python
add(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    label: Optional[int] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None
) -> int
```


Add an Embedding record to the key value store.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embedding</b>
</td>
<td>
Embedding predicted by the model.
</td>
</tr><tr>
<td>
<b>label</b>
</td>
<td>
Class numerical id. Defaults to None.
</td>
</tr><tr>
<td>
<b>data</b>
</td>
<td>
Data associated with the embedding. Defaults to None.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Associated record id.
</td>
</tr>

</table>



<h3 id="batch_add">batch_add</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L64-L90">View source</a>

```python
batch_add(
    embeddings: Sequence[<a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>],
    labels: Optional[Sequence[int]] = None,
    data: Optional[Sequence[Tensor]] = None
) -> List[int]
```


Add a set of embedding records to the key value store.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embeddings</b>
</td>
<td>
Embeddings predicted by the model.
</td>
</tr><tr>
<td>
<b>labels</b>
</td>
<td>
Class numerical ids. Defaults to None.
</td>
</tr><tr>
<td>
<b>data</b>
</td>
<td>
Data associated with the embeddings. Defaults to None.
</td>
</tr>
</table>



#### See:

add() for what a record contains.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of associated record id.
</td>
</tr>

</table>



<h3 id="batch_get">batch_get</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L106-L125">View source</a>

```python
batch_get(
    idxs: Sequence[int]
) -> Tuple[List[FloatTensor], List[Optional[int]], List[Optional[Tensor]]]
```


Get embedding records from the key value store.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>idxs</b>
</td>
<td>
ids of the records to fetch.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of records associated with the requested ids.
</td>
</tr>

</table>



<h3 id="get">get</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L92-L104">View source</a>

```python
get(
    idx: int
) -> Tuple[FloatTensor, Optional[int], Optional[Tensor]]
```


Get an embedding record from the key value store.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>idx</b>
</td>
<td>
Id of the record to fetch.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
record associated with the requested id.
</td>
</tr>

</table>



<h3 id="load">load</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L159-L176">View source</a>

```python
load(
    path: str
) -> int
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
which directory to use to store the index data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Number of records reloaded.
</td>
</tr>

</table>



<h3 id="save">save</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L131-L157">View source</a>

```python
save(
    path: str, compression: bool = True
) -> None
```


Serializes index on disk.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>path</b>
</td>
<td>
where to store the data.
</td>
</tr><tr>
<td>
<b>compression</b>
</td>
<td>
Compress index data. Defaults to True.
</td>
</tr>
</table>



<h3 id="size">size</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L127-L129">View source</a>

```python
size() -> int
```


Number of record in the key value store.


<h3 id="to_data_frame">to_data_frame</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/stores/memory_store.py#L189-L211">View source</a>

```python
to_data_frame(
    num_records: int = 0
) -> <a href="../../TFSimilarity/indexer/PandasDataFrame.md">TFSimilarity.indexer.PandasDataFrame```
</a>
```


Export data as a Pandas dataframe.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>num_records</b>
</td>
<td>
Number of records to export to the dataframe.
Defaults to 0 (unlimited).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
<b>pd.DataFrame</b>
</td>
<td>
a pandas dataframe.
</td>
</tr>
</table>





