# TFSimilarity.distances.SquaredEuclideanDistance





Compute pairwise squared Euclidean distance.

Inherits From: [`Distance`](../../TFSimilarity/distances/Distance.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.distances.SquaredEuclideanDistance()
```



<!-- Placeholder for "Used in" -->

The [Sequared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance) is
an distance that varies from 0 (similar) to infinity (dissimilar).

## Methods

<h3 id="call">call</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L152-L170">View source</a>

``<b>python
@tf.function</b>``

```python
call(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
```


Compute pairwise distances for a given batch of embeddings.


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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L34-L35">View source</a>

```python
get_config()
```





<h3 id="__call__">__call__</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L28-L29">View source</a>

```python
__call__(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
)
```


Call self as a function.




