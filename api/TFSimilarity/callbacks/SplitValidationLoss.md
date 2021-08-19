# TFSimilarity.callbacks.SplitValidationLoss





A split validation callback.

Inherits From: [`Callback`](../../TFSimilarity/callbacks/Callback.md)

```python
TFSimilarity.callbacks.SplitValidationLoss(
```

```

    known_classes: np.ndarray
)
```



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
<b>x</b>
</td>
<td>
Validation data.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
Validation labels.
</td>
</tr><tr>
<td>
<b>known_classes</b>
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
<b>x_known</b>
</td>
<td>
The set of examples from the known classes.
</td>
</tr><tr>
<td>
<b>y_known</b>
</td>
<td>
The labels associated with the known examples.
</td>
</tr><tr>
<td>
<b>x_unknown</b>
</td>
<td>
The set of examples from the unknown classes.
</td>
</tr><tr>
<td>
<b>y_unknown</b>
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
<b>x_known</b>
</td>
<td>
The set of examples from the known classes.
</td>
</tr><tr>
<td>
<b>y_known</b>
</td>
<td>
The labels associated with the known examples.
</td>
</tr><tr>
<td>
<b>x_unknown</b>
</td>
<td>
The set of examples from the unknown classes.
</td>
</tr><tr>
<td>
<b>y_unknown</b>
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






