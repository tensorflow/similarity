# TFSimilarity.layers.MetricEmbedding





This is the class from which all layers inherit.

```python
TFSimilarity.layers.MetricEmbedding(
    units: int,
    activation: Optional[Any] = None,
    use_bias: bool = True,
    kernel_initializer: Optional[Any] = glorot_uniform,
    bias_initializer: Optional[Any] = zeros,
    kernel_regularizer: Optional[Any] = None,
    bias_regularizer: Optional[Any] = None,
    activity_regularizer: Optional[Any] = None,
    kernel_constraint: Optional[Any] = None,
    bias_constraint: Optional[Any] = None,
    **kwargs
) -> None
```



<!-- Placeholder for "Used in" -->

A layer is a callable object that takes as input one or more tensors and
that outputs one or more tensors. It involves *computation*, defined
in the <b>call()</b> method, and a *state* (weight variables), defined
either in the constructor <b>__init__()</b> or in the <b>build()</b> method.

Users will just instantiate a layer and then treat it as a callable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>trainable</b>
</td>
<td>
Boolean, whether the layer's variables should be trainable.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
String name of the layer.
</td>
</tr><tr>
<td>
<b>dtype</b>
</td>
<td>
The dtype of the layer's computations and weights. Can also be a
<b>tf.keras.mixed_precision.Policy</b>, which allows the computation and weight
dtype to differ. Default of <b>None</b> means to use
<b>tf.keras.mixed_precision.global_policy()</b>, which is a float32 policy
unless set to different value.
</td>
</tr><tr>
<td>
<b>dynamic</b>
</td>
<td>
Set this to <b>True</b> if your layer should only be run eagerly, and
should not be used to generate a static computation graph.
This would be the case for a Tree-RNN or a recursive network,
for example, or generally for any layer that manipulates tensors
using Python control flow. If <b>False</b>, we assume that the layer can
safely be used to generate a static computation graph.
</td>
</tr>
</table>


We recommend that descendants of <b>Layer</b> implement the following methods:

* <b>__init__()</b>: Defines custom layer attributes, and creates layer state
  variables that do not depend on input shapes, using <b>add_weight()</b>.
* <b>build(self, input_shape)</b>: This method can be used to create weights that
  depend on the shape(s) of the input(s), using <b>add_weight()</b>. <b>__call__()</b>
  will automatically build the layer (if it has not been built yet) by
  calling <b>build()</b>.
* <b>call(self, inputs, *args, **kwargs)</b>: Called in <b>__call__</b> after making
  sure <b>build()</b> has been called. <b>call()</b> performs the logic of applying the
  layer to the input tensors (which should be passed in as argument).
  Two reserved keyword arguments you can optionally use in <b>call()</b> are:
    - <b>training</b> (boolean, whether the call is in inference mode or training
      mode). See more details in [the layer/model subclassing guide](
      https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
    - <b>mask</b> (boolean tensor encoding masked timesteps in the input, used
      in RNN layers). See more details in [the layer/model subclassing guide](
      https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
  A typical signature for this method is <b>call(self, inputs)</b>, and user could
  optionally add <b>training</b> and <b>mask</b> if the layer need them. <b>*args</b> and
  <b>**kwargs</b> is only useful for future extension when more input parameters
  are planned to be added.
* <b>get_config(self)</b>: Returns a dictionary containing the configuration used
  to initialize this layer. If the keys differ from the arguments
  in <b>__init__</b>, then override <b>from_config(self)</b> as well.
  This method is used when saving
  the layer or a model that contains this layer.

##### # # # # # # # [4. 4.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

For more information about creating layers, see the guide
- [Making new Layers and Models via subclassing](
  https://www.tensorflow.org/guide/keras/custom_layers_and_models)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
The name of the layer (string).
</td>
</tr><tr>
<td>
<b>dtype</b>
</td>
<td>
The dtype of the layer's weights.
</td>
</tr><tr>
<td>
<b>variable_dtype</b>
</td>
<td>
Alias of <b>dtype</b>.
</td>
</tr><tr>
<td>
<b>compute_dtype</b>
</td>
<td>
The dtype of the layer's computations. Layers automatically
cast inputs to this dtype which causes the computations and output to also
be in this dtype. When mixed precision is used with a
<b>tf.keras.mixed_precision.Policy</b>, this will be different than
<b>variable_dtype</b>.
</td>
</tr><tr>
<td>
<b>dtype_policy</b>
</td>
<td>
The layer's dtype policy. See the
<b>tf.keras.mixed_precision.Policy</b> documentation for details.
</td>
</tr><tr>
<td>
<b>trainable_weights</b>
</td>
<td>
List of variables to be included in backprop.
</td>
</tr><tr>
<td>
<b>non_trainable_weights</b>
</td>
<td>
List of variables that should not be
included in backprop.
</td>
</tr><tr>
<td>
<b>weights</b>
</td>
<td>
The concatenation of the lists trainable_weights and
non_trainable_weights (in this order).
</td>
</tr><tr>
<td>
<b>trainable</b>
</td>
<td>
Whether the layer should be trained (boolean), i.e. whether
its potentially-trainable weights should be returned as part of
<b>layer.trainable_weights</b>.
</td>
</tr><tr>
<td>
<b>input_spec</b>
</td>
<td>
Optional (list of) <b>InputSpec</b> object(s) specifying the
constraints on inputs that can be accepted by the layer.
</td>
</tr>
</table>



