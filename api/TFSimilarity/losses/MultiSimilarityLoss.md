# TFSimilarity.losses.MultiSimilarityLoss
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/losses/multisim_loss.py#L134-L195">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Computes the multi similarity loss in an online fashion.
Inherits From: [`MetricLoss`](../../TFSimilarity/losses/MetricLoss.md)
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.losses.MultiSimilarityLoss(
    distance: Union[<a href="../../TFSimilarity/distances/Distance.md"><code>TFSimilarity.distances.Distance</code></a>, str] = &#x27;cosine&#x27;,
    alpha: float = 1.0,
    beta: float = 20,
    epsilon: float = 0.2,
    lmda: float = 0.5,
    name: str = None
)
</code></pre>

<!-- Placeholder for "Used in" -->

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
distances between embeddings. Defaults to 'cosine'.
</td>
</tr><tr>
<td>
`alpha`
</td>
<td>
The exponential weight for the positive pairs. Increasing
alpha makes the logsumexp softmax closer to the max positive pair
distance, while decreasing it makes it closer to
max(P) + log(batch_size).
</td>
</tr><tr>
<td>
`beta`
</td>
<td>
The exponential weight for the negative pairs. Increasing
beta makes the logsumexp softmax closer to the max negative pair
distance, while decreasing it makes the softmax closer to
max(N) + log(batch_size).
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
Used to remove easy positive and negative pairs. We only
keep positives that we greater than the (smallest negative pair -
epsilon) and we only keep negatives that are less than the
(largest positive pair + epsilon).
</td>
</tr><tr>
<td>
`lmda`
</td>
<td>
Used to weight the distance. Below this distance, negatives
are up weighted and positives are down weighted. Similarly, above
this distance negatives are down weighted and positive are up
weighted.
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


