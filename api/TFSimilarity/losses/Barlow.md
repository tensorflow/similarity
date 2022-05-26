# TFSimilarity.losses.Barlow





Computes the Barlow Loss between two batches of embeddings.

```python
TFSimilarity.losses.Barlow(
    lambda_: float = 0.005,
    margin: float = 1e-12,
    reduction: Callable = tf.keras.losses.Reduction.AUTO,
    name: Optional[str] = None,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

Reference

Zbontar, Jure, et al.
"Barlow Twins: Self-Supervised Learning via Redundancy Reduction."
https://arxiv.org/abs/2103.03230

#### Standalone usage:



```
>>> loss = tensorflow_similarity.losses.Barlow()
>>> za = tf.random.uniform(shape=[4, 16])
>>> zb = tf.random.uniform(shape=[4, 16])
>>> loss(za, zb)
<tf.Tensor: shape=(), dtype=float32, numpy=22.144062>
```

Usage with the <b>compile()</b> API:
```python
model.compile(optimizer='sgd', loss=tensorflow_similarity.losses.Barlow())
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>reduction</b>
</td>
<td>
Type of <b>tf.keras.losses.Reduction</b> to apply to
loss. Default value is <b>AUTO</b>. <b>AUTO</b> indicates that the reduction
option will be determined by the usage context. For almost all cases
this defaults to <b>SUM_OVER_BATCH_SIZE</b>. When used with
<b>tf.distribute.Strategy</b>, outside of built-in training loops such as
<b>tf.keras</b> <b>compile</b> and <b>fit</b>, using <b>AUTO</b> or <b>SUM_OVER_BATCH_SIZE</b>
will raise an error. Please see this custom training [tutorial](
  https://www.tensorflow.org/tutorials/distribute/custom_training) for
    more details.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
Optional name for the instance.
</td>
</tr>
</table>



## Methods

<h3 id="from_config">from_config</h3>

``<b>python
@classmethod</b>``

```python
from_config(
    config
)
```


Instantiates a <b>Loss</b> from its config (output of <b>get_config()</b>).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>config</b>
</td>
<td>
Output of <b>get_config()</b>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Loss</b> instance.
</td>
</tr>

</table>



<h3 id="get_config">get_config</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/barlow.py#L95-L101">View source</a>

```python
get_config() -> Dict[str, Any]
```


Returns the config dictionary for a <b>Loss</b> instance.


<h3 id="off_diagonal">off_diagonal</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/barlow.py#L103-L108">View source</a>

```python
off_diagonal(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
```





<h3 id="standardize_columns">standardize_columns</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/barlow.py#L110-L115">View source</a>

```python
standardize_columns(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>
```





<h3 id="__call__">__call__</h3>

```python
__call__(
    y_true, y_pred, sample_weight=None
)
```


Invokes the <b>Loss</b> instance.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>y_true</b>
</td>
<td>
Ground truth values. shape = <b>[batch_size, d0, .. dN]</b>, except
sparse loss functions such as sparse categorical crossentropy where
shape = <b>[batch_size, d0, .. dN-1]</b>
</td>
</tr><tr>
<td>
<b>y_pred</b>
</td>
<td>
The predicted values. shape = <b>[batch_size, d0, .. dN]</b>
</td>
</tr><tr>
<td>
<b>sample_weight</b>
</td>
<td>
Optional <b>sample_weight</b> acts as a coefficient for the
loss. If a scalar is provided, then the loss is simply scaled by the
given value. If <b>sample_weight</b> is a tensor of size <b>[batch_size]</b>, then
the total loss for each sample of the batch is rescaled by the
corresponding element in the <b>sample_weight</b> vector. If the shape of
<b>sample_weight</b> is <b>[batch_size, d0, .. dN-1]</b> (or can be broadcasted to
this shape), then each loss element of <b>y_pred</b> is scaled
by the corresponding value of <b>sample_weight</b>. (Note on<b>dN-1</b>: all loss
  functions reduce by 1 dimension, usually axis=-1.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Weighted loss float <b>Tensor</b>. If <b>reduction</b> is <b>NONE</b>, this has
shape <b>[batch_size, d0, .. dN-1]</b>; otherwise, it is scalar. (Note <b>dN-1</b>
because all loss functions reduce by 1 dimension, usually axis=-1.)
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<b>ValueError</b>
</td>
<td>
If the shape of <b>sample_weight</b> is invalid.
</td>
</tr>
</table>





