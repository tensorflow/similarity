# TFSimilarity.losses.CircleLoss
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/circle_loss.py#L130-L181">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the CircleLoss
Inherits From: [`MetricLoss`](../../TFSimilarity/losses/MetricLoss.md)
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.losses.CircleLoss(
    distance: Union[<a href="../../TFSimilarity/distances/Distance.md"><code>TFSimilarity.distances.Distance</code></a>, str] = &#x27;cosine&#x27;,
    gamma: float = 80.0,
    margin: float = 0.4,
    name: str = None
)
</code></pre>

<!-- Placeholder for "Used in" -->
Circle Loss: A Unified Perspective of Pair Similarity Optimization
https://arxiv.org/abs/2002.10857
The original paper used cosine similarity while this loss has been
modified to work with cosine distance.
`y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
It's values represent the classes associated with the examples as
**integer  values**.
`y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
last layer of your model to ensure your model output is properly
normalized.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`distance`
</td>
<td>
Which distance function to use to compute the pairwise
distances between embeddings. The distance is expected to be
between [0, 2]. Defaults to 'cosine'.
</td>
</tr><tr>
<td>
`gamma`
</td>
<td>
Scaling term. Defaults to 80. Note: Large values cause the
LogSumExp to return the Max pair and reduces the weighted mixing
of all pairs. Should be hypertuned.
</td>
</tr><tr>
<td>
`margin`
</td>
<td>
Used to weight the distance. Below this distance, negatives
are up weighted and positives are down weighted. Similarly, above
this distance negatives are down weighted and positive are up
weighted. Defaults to 0.4.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Loss name. Defaults to None.
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
Instantiates a `Loss` from its config (output of `get_config()`).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`config`
</td>
<td>
Output of `get_config()`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Loss` instance.
</td>
</tr>
</table>

<h3 id="get_config"><code>get_config</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/metric_loss.py#L57-L70">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>
Returns the config dictionary for a `Loss` instance.

<h3 id="__call__"><code>__call__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true, y_pred, sample_weight=None
)
</code></pre>
Invokes the `Loss` instance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`y_true`
</td>
<td>
Ground truth values. shape = `[batch_size, d0, .. dN]`, except
sparse loss functions such as sparse categorical crossentropy where
shape = `[batch_size, d0, .. dN-1]`
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values. shape = `[batch_size, d0, .. dN]`
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional `sample_weight` acts as a coefficient for the
loss. If a scalar is provided, then the loss is simply scaled by the
given value. If `sample_weight` is a tensor of size `[batch_size]`, then
the total loss for each sample of the batch is rescaled by the
corresponding element in the `sample_weight` vector. If the shape of
`sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
this shape), then each loss element of `y_pred` is scaled
by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
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
Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
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
`ValueError`
</td>
<td>
If the shape of `sample_weight` is invalid.
</td>
</tr>
</table>


