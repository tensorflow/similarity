# TFSimilarity.layers.MetricEmbedding





Just your regular densely-connected NN layer.

```python
TFSimilarity.layers.MetricEmbedding(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=glorot_uniform,
    bias_initializer=zeros,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

<b>Dense</b> implements the operation:
<b>output = activation(dot(input, kernel) + bias)</b>
where <b>activation</b> is the element-wise activation function
passed as the <b>activation</b> argument, <b>kernel</b> is a weights matrix
created by the layer, and <b>bias</b> is a bias vector created by the layer
(only applicable if <b>use_bias</b> is <b>True</b>). These are all attributes of
<b>Dense</b>.

Note: If the input to the layer has a rank greater than 2, then <b>Dense</b>
computes the dot product between the <b>inputs</b> and the <b>kernel</b> along the
last axis of the <b>inputs</b> and axis 0 of the <b>kernel</b> (using <b>tf.tensordot</b>).
For example, if input has dimensions <b>(batch_size, d0, d1)</b>,
then we create a <b>kernel</b> with shape <b>(d1, units)</b>, and the <b>kernel</b> operates
along axis 2 of the <b>input</b>, on every sub-tensor of shape <b>(1, 1, d1)</b>
(there are <b>batch_size * d0</b> such sub-tensors).
The output in this case will have shape <b>(batch_size, d0, units)</b>.

Besides, layer attributes cannot be modified after the layer has been called
once (except the <b>trainable</b> attribute).
When a popular kwarg <b>input_shape</b> is passed, then keras will create
an input layer to insert before the current layer. This can be treated
equivalent to explicitly defining an <b>InputLayer</b>.

##### # # # # the size of the input anymore:
>>> model.add(tf.keras.layers.Dense(32))
>>> model.output_shape
(None, 32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>units</b>
</td>
<td>
Positive integer, dimensionality of the output space.
</td>
</tr><tr>
<td>
<b>activation</b>
</td>
<td>
Activation function to use.
If you don't specify anything, no activation is applied
(ie. "linear" activation: <b>a(x) = x</b>).
</td>
</tr><tr>
<td>
<b>use_bias</b>
</td>
<td>
Boolean, whether the layer uses a bias vector.
</td>
</tr><tr>
<td>
<b>kernel_initializer</b>
</td>
<td>
Initializer for the <b>kernel</b> weights matrix.
</td>
</tr><tr>
<td>
<b>bias_initializer</b>
</td>
<td>
Initializer for the bias vector.
</td>
</tr><tr>
<td>
<b>kernel_regularizer</b>
</td>
<td>
Regularizer function applied to
the <b>kernel</b> weights matrix.
</td>
</tr><tr>
<td>
<b>bias_regularizer</b>
</td>
<td>
Regularizer function applied to the bias vector.
</td>
</tr><tr>
<td>
<b>activity_regularizer</b>
</td>
<td>
Regularizer function applied to
the output of the layer (its "activation").
</td>
</tr><tr>
<td>
<b>kernel_constraint</b>
</td>
<td>
Constraint function applied to
the <b>kernel</b> weights matrix.
</td>
</tr><tr>
<td>
<b>bias_constraint</b>
</td>
<td>
Constraint function applied to the bias vector.
</td>
</tr>
</table>



#### Input shape:

N-D tensor with shape: <b>(batch_size, ..., input_dim)</b>.
The most common situation would be
a 2D input with shape <b>(batch_size, input_dim)</b>.



#### Output shape:

N-D tensor with shape: <b>(batch_size, ..., units)</b>.
For instance, for a 2D input with shape <b>(batch_size, input_dim)</b>,
the output would have shape <b>(batch_size, units)</b>.


