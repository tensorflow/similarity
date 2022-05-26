# TFSimilarity.indexer.Lookup





Metadata associated with a query match.


```python
TFSimilarity.indexer.Lookup(
    rank: int,
    distance: float,
    label: Optional[int] = dataclasses.field(default=None),
    embedding: Optional[np.ndarray] = dataclasses.field(default=None),
)
```



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>rank</b>
</td>
<td>
Rank of the match with respect to the query distance.
</td>
</tr><tr>
<td>
<b>distance</b>
</td>
<td>
The distance from the match to the query.
</td>
</tr><tr>
<td>
<b>label</b>
</td>
<td>
The label associated with the match. Default None.
</td>
</tr><tr>
<td>
<b>embedding</b>
</td>
<td>
The embedded match vector. Default None.
</td>
</tr><tr>
<td>
<b>data</b>
</td>
<td>
The original Tensor representation of the match result.
Default None.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__">__eq__</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/types.py#L114-L128">View source</a>

```python
__eq__(
    other
) -> bool
```


Return self==value.






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
data<a id="data"></a>
</td>
<td>
<b>None</b>
</td>
</tr><tr>
<td>
embedding<a id="embedding"></a>
</td>
<td>
<b>None</b>
</td>
</tr><tr>
<td>
label<a id="label"></a>
</td>
<td>
<b>None</b>
</td>
</tr>
</table>

