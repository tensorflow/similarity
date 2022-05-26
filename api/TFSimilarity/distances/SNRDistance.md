# TFSimilarity.distances.SNRDistance





Computes pairwise SNR distances between embeddings.

Inherits From: [`Distance`](../../TFSimilarity/distances/Distance.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.distances.SNRDistance()
```



<!-- Placeholder for "Used in" -->

The [Signal-to-Noise Ratio distance](https://arxiv.org/abs/1904.02616)
is the ratio of noise variance to the feature variance.

## Methods

<h3 id="call">call</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L243-L267">View source</a>

``<b>python
@tf.function</b>``

```python
call(
    query_embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    key_embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
```


Compute pairwise snr distances for a given batch of embeddings.
SNR(i, j): anchor i and compared feature j
SNR(i,j) may not be equal to SNR(j, i)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>query_embeddings</b>
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

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L51-L52">View source</a>

```python
get_config()
```





<h3 id="__call__">__call__</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L45-L46">View source</a>

```python
__call__(
    query_embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    key_embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
)
```


Call self as a function.




