# TFSimilarity.distances.InnerProductSimilarity





Compute the pairwise inner product between embeddings.

Inherits From: [`Distance`](../../TFSimilarity/distances/Distance.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.distances.InnerProductSimilarity()
```



<!-- Placeholder for "Used in" -->

The [Inner product](https://en.wikipedia.org/wiki/Inner_product_space) is
a measure of similarity where the more similar vectors have the largest
values.

NOTE! This is not a distance and is likely not what you want to use with
the built in losses. At the very least this will flip the sign on the
margin in many of the losses. This is likely meant to be used with custom
loss functions that expect a similarity instead of a distance.

## Methods

<h3 id="call">call</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L71-L84">View source</a>

``<b>python
@tf.function</b>``

```python
call(
    embeddings: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>
) -> <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>
```


Compute pairwise similarities for a given batch of embeddings.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>embeddings</b>
</td>
<td>
Embeddings to compute the pairwise one.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
<b>FloatTensor</b>
</td>
<td>
Pairwise distance tensor.
</td>
</tr>
</table>



<h3 id="get_config">get_config</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L50-L51">View source</a>

```python
get_config()
```





<h3 id="__call__">__call__</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L44-L45">View source</a>

```python
__call__(
    embeddings: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>
)
```


Call self as a function.




