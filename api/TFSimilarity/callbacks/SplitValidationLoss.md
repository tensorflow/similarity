
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.callbacks.SplitValidationLoss" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>
# TFSimilarity.callbacks.SplitValidationLoss
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/callbacks.py#L100-L166">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

A split validation callback.
Inherits From: [`Callback`](../../TFSimilarity/callbacks/Callback.md)
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.callbacks.SplitValidationLoss(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    known_classes: np.ndarray
)
</code></pre>

<!-- Placeholder for "Used in" -->
This callback will split the validation data into two sets.
    1) The set of classes seen during training.
    2) The set of classes not seen during training.
The callback will then compute a separate validation for each split.
This is useful for separately tracking the validation loss on the seen and
unseen classes and may provide insight into how well the embedding will
generalize to new classes.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`x`
</td>
<td>
Validation data.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Validation labels.
</td>
</tr><tr>
<td>
`known_classes`
</td>
<td>
The set of classes seen during training.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`x_known`
</td>
<td>
The set of examples from the known classes.
</td>
</tr><tr>
<td>
`y_known`
</td>
<td>
The labels associated with the known examples.
</td>
</tr><tr>
<td>
`x_unknown`
</td>
<td>
The set of examples from the unknown classes.
</td>
</tr><tr>
<td>
`y_unknown`
</td>
<td>
The labels associated with the unknown examples.
</td>
</tr>
</table>


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`x_known`
</td>
<td>
The set of examples from the known classes.
</td>
</tr><tr>
<td>
`y_known`
</td>
<td>
The labels associated with the known examples.
</td>
</tr><tr>
<td>
`x_unknown`
</td>
<td>
The set of examples from the unknown classes.
</td>
</tr><tr>
<td>
`y_unknown`
</td>
<td>
The labels associated with the unknown examples.
</td>
</tr>
</table>

## Methods
<h3 id="set_model"><code>set_model</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>


<h3 id="set_params"><code>set_params</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>



