# TFSimilarity.losses.TripletLoss





Computes the triplet loss in an online fashion.

Inherits From: [`MetricLoss`](../../TFSimilarity/losses/MetricLoss.md)

```python
TFSimilarity.losses.TripletLoss(
```

    positive_mining_strategy: str = hard,
    negative_mining_strategy: str = semi-hard,
    soft_margin: bool = (False),
    margin: float = 1.0,
    name: str = None
)
```



<!-- Placeholder for "Used in" -->

This loss encourages the positive distances between a pair of embeddings
with the same labels to be smaller than the minimum negative distances
between pair of embeddings of different labels.
See: https://arxiv.org/abs/1503.03832 for the original paper.


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
Which distance function to use to compute
the pairwise distances between embeddings. Defaults to 'cosine'.
</td>
</tr><tr>
<td>
<b>positive_mining_strategy</b>
</td>
<td>
What mining strategy to
use to select embedding from the same class. Defaults to 'hard'.
</td>
</tr><tr>
<td>
<b>available</b>
</td>
<td>
<i>'easy', 'hard'</i>
</td>
</tr><tr>
<td>
<b>negative_mining_strategy</b>
</td>
<td>
What mining strategy to
use for select the embedding from the different class.
Defaults to 'semi-hard'. Available: <i>'hard', 'semi-hard', 'easy'</i>
</td>
</tr><tr>
<td>
<b>soft_margin</b>
</td>
<td>
- [description]. Defaults to True.
Use a soft margin instead of an explicit one.
</td>
</tr><tr>
<td>
<b>margin</b>
</td>
<td>
Use an explicit value for the margin
term. Defaults to 1.0.
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<b>ValueError</b>
</td>
<td>
Invalid positive mining strategy.
</td>
</tr><tr>
<td>
<b>ValueError</b>
</td>
<td>
Invalid negative mining strategy.
</td>
</tr><tr>
<td>
<b>ValueError</b>
</td>
<td>
Margin value is not used when soft_margin is set
to True.
</td>
</tr>
</table>



## Methods

<h3 id="from_config"><code>from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

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



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/metric_loss.py#L57-L70">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>

Returns the config dictionary for a <b>Loss</b> instance.


<h3 id="__call__"><code>__call__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true, y_pred, sample_weight=None
)
</code></pre>

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





