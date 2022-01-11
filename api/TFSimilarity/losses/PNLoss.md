# TFSimilarity.losses.PNLoss





Computes the PN loss in an online fashion.

Inherits From: [`MetricLoss`](../../TFSimilarity/losses/MetricLoss.md)

```python
TFSimilarity.losses.PNLoss(
    positive_mining_strategy: str = hard,
    negative_mining_strategy: str = semi-hard,
    soft_margin: bool = False,
    margin: float = 1.0,
    name: str = PNLoss,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

This loss encourages the positive distances between a pair of embeddings
with the same labels to be smaller than the minimum negative distances
between pair of embeddings of different labels. Additionally, both the
anchor and the positive embeddings are encouraged to be far from the
negative embeddings. This is accomplished by taking the
min(pos_neg_dist, anchor_neg_dist) and using that as the negative distance
in the triplet loss.

#### See PN Loss Ivis:


Szubert, B., Cole, J.E., Monaco, C. et al.
Structure-preserving visualisation of high dimensional single-cell dataset
Sci Rep 9, 8914 (2019). https://doi.org/10.1038/s41598-019-45301-0

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
Loss name. Defaults to PNLoss.
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/metric_loss.py#L57-L71">View source</a>

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





