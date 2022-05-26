# TFSimilarity.losses.MetricLoss





Wraps a loss function in the `Loss` class.

```python
TFSimilarity.losses.MetricLoss(
    fn: Callable,
    reduction: Callable = tf.keras.losses.Reduction.AUTO,
    name: Optional[str] = None,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>fn</b>
</td>
<td>
The loss function to wrap, with signature `fn(y_true, y_pred,
**kwargs)`.
</td>
</tr><tr>
<td>
<b>reduction</b>
</td>
<td>
(Optional) Type of <b>tf.keras.losses.Reduction</b> to apply to
loss. Default value is <b>AUTO</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
(Optional) name for the loss.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
The keyword arguments that are passed on to <b>fn</b>.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/metric_loss.py#L61-L75">View source</a>

```python
get_config() -> Dict[str, Any]
```


Contains the loss configuration.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Python dict containing the configuration of the loss.
</td>
</tr>

</table>



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





