# TFSimilarity.indexer.MemoryTable
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L9-L182">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Efficient in-memory dataset table
Inherits From: [`Table`](../../TFSimilarity/indexer/Table.md), [`ABC`](../../TFSimilarity/distances/ABC.md)
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.tables.MemoryTable`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.indexer.MemoryTable() -> None
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods
<h3 id="add"><code>add</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L21-L42">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add(
    embedding: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    label: Optional[int] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>] = None
) -> int
</code></pre>
Add an Embedding record to the table.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embedding`
</td>
<td>
Embedding predicted by the model.
</td>
</tr><tr>
<td>
`label`
</td>
<td>
Class numerical id. Defaults to None.
</td>
</tr><tr>
<td>
`data`
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

<h3 id="batch_add"><code>batch_add</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L44-L70">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_add(
    embeddings: List[<a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>],
    labels: List[Optional[int]] = None,
    data: List[Optional[Tensor]] = None
) -> List[int]
</code></pre>
Add a set of embedding records to the table.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embeddings`
</td>
<td>
Embeddings predicted by the model.
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
Class numerical ids. Defaults to None.
</td>
</tr><tr>
<td>
`data`
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

<h3 id="batch_get"><code>batch_get</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L86-L105">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_get(
    idxs: List[int]
) -> Tuple[List[FloatTensor], List[Optional[int]], List[Optional[Tensor]]]
</code></pre>
Get embedding records from the table

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`idxs`
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

<h3 id="get"><code>get</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L72-L84">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    idx: int
) -> Tuple[FloatTensor, Optional[int], Optional[Tensor]]
</code></pre>
Get an embedding record from the table

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`idx`
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

<h3 id="load"><code>load</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L130-L147">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load(
    path: str
) -> int
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

<h3 id="save"><code>save</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L111-L128">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    path: str,
    compression: bool = (True)
) -> None
</code></pre>
Serializes index on disk.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`path`
</td>
<td>
where to store the data.
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
Compress index data. Defaults to True.
</td>
</tr>
</table>

<h3 id="size"><code>size</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L107-L109">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>size() -> int
</code></pre>
Number of record in the table.

<h3 id="to_data_frame"><code>to_data_frame</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/tables/memory_table.py#L160-L182">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_data_frame(
    num_records: int = 0
) -> <a href="../../TFSimilarity/indexer/PandasDataFrame.md"><code>TFSimilarity.indexer.PandasDataFrame</code></a>
</code></pre>
Export data as a Pandas dataframe.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`num_records`
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
`pd.DataFrame`
</td>
<td>
a pandas dataframe.
</td>
</tr>
</table>


