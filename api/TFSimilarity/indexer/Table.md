# TFSimilarity.indexer.Table





Helper class that provides a standard way to create an ABC using

Inherits From: [`ABC`](../../TFSimilarity/distances/ABC.md)

<!-- Placeholder for "Used in" -->
inheritance.

## Methods

<h3 id="add">add</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L8-L24">View source</a>

```python
add(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    label: Optional[int] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None
) -> int
```


Add an Embedding record to the table.


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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L25-L44">View source</a>

```python
batch_add(
    embeddings: List[<a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>],
    labels: List[Optional[int]] = None,
    data: List[Optional[Tensor]] = None
) -> List[int]
```


Add a set of embedding records to the table.


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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L59-L70">View source</a>

```python
batch_get(
    idxs: List[int]
) -> Tuple[List[FloatTensor], List[Optional[int]], List[Optional[Tensor]]]
```


Get embedding records from the table


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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L46-L57">View source</a>

```python
get(
    idx: int
) -> Tuple[FloatTensor, Optional[int], Optional[Tensor]]
```


Get an embedding record from the table


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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L85-L94">View source</a>

```python
load(
    path: str
) -> int
```


Load index on disk


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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L76-L83">View source</a>

```python
save(
    path: str,
    compression: bool = (True)
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
Directory where to store the data.
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L72-L74">View source</a>

```python
size() -> int
```


Number of record in the table.


<h3 id="to_data_frame">to_data_frame</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/table.py#L96-L106">View source</a>

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





