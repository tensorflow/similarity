# TFSimilarity.losses.CircleLoss





Computes the CircleLoss

Inherits From: [`MetricLoss`](../../TFSimilarity/losses/MetricLoss.md)

```python
TFSimilarity.losses.CircleLoss(
    gamma: float = 80.0,
    margin: float = 0.4,
    name: str = None
)
```



<!-- Placeholder for "Used in" -->

Circle Loss: A Unified Perspective of Pair Similarity Optimization
https://arxiv.org/abs/2002.10857

The original paper used cosine similarity while this loss has been
modified to work with cosine distance.

<b>y_true</b> must be  a 1-D integer <b>Tensor</b> of shape (batch_size,).
It's values represent the classes associated with the examples as
**integer  values**.

<b>y_pred</b> must be 2-D float <b>Tensor</b>  of L2 normalized embedding vectors.
you can use the layer <b>tensorflow_similarity.layers.L2Embedding()</b> as the
last layer of your model to ensure your model output is properly
normalized.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>distance</b>
</td>
<td>
Which distance function to use to compute the pairwise
distances between embeddings. The distance is expected to be
between [0, 2]. Defaults to 'cosine'.
</td>
</tr><tr>
<td>
<b>gamma</b>
</td>
<td>
Scaling term. Defaults to 80. Note: Large values cause the
LogSumExp to return the Max pair and reduces the weighted mixing
of all pairs. Should be hypertuned.
</td>
</tr><tr>
<td>
<b>margin</b>
</td>
<td>
Used to weight the distance. Below this distance, negatives
are up weighted and positives are down weighted. Similarly, above
this distance negatives are down weighted and positive are up
weighted. Defaults to 0.4.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
Loss name. Defaults to None.
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/metric_loss.py#L57-L70">View source</a>

```python
get_config() -> Dict[str, Any]
```


Returns the config dictionary for a <b>Loss</b> instance.


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





