# TFSimilarity.layers.Layer






This is the class from which all layers inherit.

```python
TFSimilarity.layers.Layer(
    trainable=(True), name=None, dtype=None, dynamic=(False), **kwargs
)
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

##### # # # # # # # # # # # Weight regularization.
>>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
>>> model.losses
- [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
```
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
List of metrics added using the <b>add_metric()</b> API.


```
>>> input = tf.keras.layers.Input(shape=(3,))
>>> d = tf.keras.layers.Dense(2)
>>> output = d(input)
>>> d.add_metric(tf.reduce_max(output), name='max')
>>> d.add_metric(tf.reduce_min(output), name='min')
>>> [m.name for m in d.metrics]
- ['max', 'min']
```
</td>
</tr><tr>
<td>
<b>output</b>
</td>
<td>
Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.
</td>
</tr><tr>
<td>
<b>supports_masking</b>
</td>
<td>
Whether this layer supports computing a mask using <b>compute_mask</b>.
</td>
</tr>
</table>



## Methods

<h3 id="add_loss">add_loss</h3>

```python
add_loss(
    losses, **kwargs
)
```


Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be dependent
on the inputs passed when calling a layer. Hence, when reusing the same
layer on different inputs <b>a</b> and <b>b</b>, some entries in <b>layer.losses</b> may
be dependent on <b>a</b> and some on <b>b</b>. This method automatically keeps track
of dependencies.

This method can be used inside a subclassed layer or model's <b>call</b>
function, in which case <b>losses</b> should be a Tensor or list of Tensors.

#### Example:



```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any loss Tensors passed to this Model must
be symbolic and be able to be traced back to the model's <b>Input</b>s. These
losses become part of the model's topology and are tracked in <b>get_config</b>.

##### Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

If this is not the case for your loss (if, for example, your loss references
a <b>Variable</b> of one of the model's layers), you can wrap your loss in a
zero-argument lambda. These losses are not tracked as part of the model's
topology since they can't be serialized.

##### Weight regularization.
model.add_loss(lambda: tf.reduce_mean(d.kernel))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>losses</b>
</td>
<td>
Loss tensor, or list/tuple of tensors. Rather than tensors, losses
may also be zero-argument callables which create a loss tensor.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
  inputs - Deprecated, will be automatically inferred.
</td>
</tr>
</table>



<h3 id="add_metric">add_metric</h3>

```python
add_metric(
    value, name=None, **kwargs
)
```


Adds metric tensor to the layer.

This method can be used inside the <b>call()</b> method of a subclassed layer
or model.

```python
class MyMetricLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

  def call(self, inputs):
    self.add_metric(self.mean(inputs))
    self.add_metric(tf.reduce_sum(inputs), name='metric_2')
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any tensor passed to this Model must
be symbolic and be able to be traced back to the model's <b>Input</b>s. These
metrics become part of the model's topology and are tracked when you
save the model via <b>save()</b>.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(math_ops.reduce_sum(x), name='metric_1')
```

Note: Calling <b>add_metric()</b> with the result of a metric object on a
Functional Model, as shown in the example below, is not supported. This is
because we cannot trace the metric result tensor back to the model's inputs.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>value</b>
</td>
<td>
Metric tensor.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
String metric name.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
<b>aggregation</b> - When the <b>value</b> tensor provided is not the result of
calling a <b>keras.Metric</b> instance, it will be aggregated by default
using a <b>keras.Metric.Mean</b>.
</td>
</tr>
</table>



<h3 id="add_weight">add_weight</h3>

```python
add_weight(
    name=None, shape=None, dtype=None, initializer=None, regularizer=None,
    trainable=None, constraint=None, use_resource=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE, **kwargs
)
```


Adds a new variable to the layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
Variable name.
</td>
</tr><tr>
<td>
<b>shape</b>
</td>
<td>
Variable shape. Defaults to scalar if unspecified.
</td>
</tr><tr>
<td>
<b>dtype</b>
</td>
<td>
The type of the variable. Defaults to <b>self.dtype</b>.
</td>
</tr><tr>
<td>
<b>initializer</b>
</td>
<td>
Initializer instance (callable).
</td>
</tr><tr>
<td>
<b>regularizer</b>
</td>
<td>
Regularizer instance (callable).
</td>
</tr><tr>
<td>
<b>trainable</b>
</td>
<td>
Boolean, whether the variable should be part of the layer's
"trainable_variables" (e.g. variables, biases)
or "non_trainable_variables" (e.g. BatchNorm mean and variance).
Note that <b>trainable</b> cannot be <b>True</b> if <b>synchronization</b>
is set to <b>ON_READ</b>.
</td>
</tr><tr>
<td>
<b>constraint</b>
</td>
<td>
Constraint instance (callable).
</td>
</tr><tr>
<td>
<b>use_resource</b>
</td>
<td>
Whether to use <b>ResourceVariable</b>.
</td>
</tr><tr>
<td>
<b>synchronization</b>
</td>
<td>
Indicates when a distributed a variable will be
aggregated. Accepted values are constants defined in the class
<b>tf.VariableSynchronization</b>. By default the synchronization is set to
<b>AUTO</b> and the current <b>DistributionStrategy</b> chooses
when to synchronize. If <b>synchronization</b> is set to <b>ON_READ</b>,
<b>trainable</b> must not be set to <b>True</b>.
</td>
</tr><tr>
<td>
<b>aggregation</b>
</td>
<td>
Indicates how a distributed variable will be aggregated.
Accepted values are constants defined in the class
<b>tf.VariableAggregation</b>.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Additional keyword arguments. Accepted values are <b>getter</b>,
<b>collections</b>, <b>experimental_autocast</b> and <b>caching_device</b>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The variable created.
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
When giving unsupported dtype and no initializer or when
trainable has been set to True with synchronization set as <b>ON_READ</b>.
</td>
</tr>
</table>



<h3 id="build">build</h3>

```python
build(
    input_shape
)
```


Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of <b>Layer</b> or <b>Model</b>
can override if they need a state-creation step in-between
layer instantiation and layer call.

This is typically used to create the weights of <b>Layer</b> subclasses.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>input_shape</b>
</td>
<td>
Instance of <b>TensorShape</b>, or list of instances of
<b>TensorShape</b> if the layer expects a list of inputs
(one instance per input).
</td>
</tr>
</table>



<h3 id="call">call</h3>

```python
call(
    inputs, *args, **kwargs
)
```


This is where the layer's logic lives.

Note here that <b>call()</b> method in <b>tf.keras</b> is little bit different
from <b>keras</b> API. In <b>keras</b> API, you can pass support masking for
layers as additional arguments. Whereas <b>tf.keras</b> has <b>compute_mask()</b>
method to support masking.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>inputs</b>
</td>
<td>
Input tensor, or dict/list/tuple of input tensors.
The first positional <b>inputs</b> argument is subject to special rules:
- <b>inputs</b> must be explicitly passed. A layer cannot have zero
  arguments, and <b>inputs</b> cannot be provided via the default value
  of a keyword argument.
- NumPy array or Python scalar values in <b>inputs</b> get cast as tensors.
- Keras mask metadata is only collected from <b>inputs</b>.
- Layers are built (<b>build(input_shape)</b> method)
  using shape info from <b>inputs</b> only.
- <b>input_spec</b> compatibility is only checked against <b>inputs</b>.
- Mixed precision input casting is only applied to <b>inputs</b>.
  If a layer has tensor arguments in <b>*args</b> or <b>**kwargs</b>, their
  casting behavior in mixed precision should be handled manually.
- The SavedModel input specification is generated using <b>inputs</b> only.
- Integration with various ecosystem packages like TFMOT, TFLite,
  TF.js, etc is only supported for <b>inputs</b> and not for tensors in
  positional and keyword arguments.
</td>
</tr><tr>
<td>
<b>*args</b>
</td>
<td>
Additional positional arguments. May contain tensors, although
this is not recommended, for the reasons above.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Additional keyword arguments. May contain tensors, although
this is not recommended, for the reasons above.
The following optional keyword arguments are reserved:
- <b>training</b>: Boolean scalar tensor of Python boolean indicating
  whether the <b>call</b> is meant for training or inference.
- <b>mask</b>: Boolean input mask. If the layer's <b>call()</b> method takes a
  <b>mask</b> argument, its default value will be set to the mask generated
  for <b>inputs</b> by the previous layer (if <b>input</b> did come from a layer
  that generated a corresponding mask, i.e. if it came from a Keras
  layer with masking support).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tensor or list/tuple of tensors.
</td>
</tr>

</table>



<h3 id="compute_mask">compute_mask</h3>

```python
compute_mask(
    inputs, mask=None
)
```


Computes an output mask tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>inputs</b>
</td>
<td>
Tensor or list of tensors.
</td>
</tr><tr>
<td>
<b>mask</b>
</td>
<td>
Tensor or list of tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a tensor (or list of tensors,
one per output tensor of the layer).
</td>
</tr>

</table>



<h3 id="compute_output_shape">compute_output_shape</h3>

```python
compute_output_shape(
    input_shape
)
```


Computes the output shape of the layer.

If the layer has not been built, this method will call <b>build</b> on the
layer. This assumes that the layer will later be used with inputs that
match the input shape provided here.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>input_shape</b>
</td>
<td>
Shape tuple (tuple of integers)
or list of shape tuples (one per output tensor of the layer).
Shape tuples can include None for free dimensions,
instead of an integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An input shape tuple.
</td>
</tr>

</table>



<h3 id="compute_output_signature">compute_output_signature</h3>

```python
compute_output_signature(
    input_signature
)
```


Compute the output tensor signature of the layer based on the inputs.

Unlike a TensorShape object, a TensorSpec object contains both shape
and dtype information for a tensor. This method allows layers to provide
output dtype information if it is different from the input dtype.
For any layer that doesn't implement this function,
the framework will fall back to use <b>compute_output_shape</b>, and will
assume that the output dtype matches the input dtype.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>input_signature</b>
</td>
<td>
Single TensorSpec or nested structure of TensorSpec
objects, describing a candidate input for the layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Single TensorSpec or nested structure of TensorSpec objects, describing
how the layer would transform the provided input.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<b>TypeError</b>
</td>
<td>
If input_signature contains a non-TensorSpec object.
</td>
</tr>
</table>



<h3 id="count_params">count_params</h3>

```python
count_params()
```


Count the total number of scalars composing the weights.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer count.
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
if the layer isn't yet built
(in which case its weights aren't yet defined).
</td>
</tr>
</table>



<h3 id="from_config">from_config</h3>

``<b>python
@classmethod</b>``

```python
from_config(
    config
)
```


Creates a layer from its config.

This method is the reverse of <b>get_config</b>,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by <b>set_weights</b>).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>config</b>
</td>
<td>
A Python dictionary, typically the
output of get_config.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A layer instance.
</td>
</tr>

</table>



<h3 id="get_config">get_config</h3>

```python
get_config()
```


Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by <b>Network</b> (one layer of abstraction above).

Note that <b>get_config()</b> does not guarantee to return a fresh copy of dict
every time it is called. The callers should make a copy of the returned dict
if they want to modify it.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary.
</td>
</tr>

</table>



<h3 id="get_weights">get_weights</h3>

```python
get_weights()
```


Returns the current weights of the layer, as NumPy arrays.

The weights of a layer represent the state of the layer. This function
returns both trainable and non-trainable weight values associated with this
layer as a list of NumPy arrays, which can in turn be used to load state
into similarly parameterized layers.

For example, a <b>Dense</b> layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
<b>Dense</b> layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
- [array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
- [array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
- [array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Weights values as a list of NumPy arrays.
</td>
</tr>

</table>



<h3 id="set_weights">set_weights</h3>

```python
set_weights(
    weights
)
```


Sets the weights of the layer, from NumPy arrays.

The weights of a layer represent the state of the layer. This function
sets the weight values from numpy arrays. The weight values should be
passed in the order they are created by the layer. Note that the layer's
weights must be instantiated before calling this function, by calling
the layer.

For example, a <b>Dense</b> layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
<b>Dense</b> layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
- [array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
- [array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
- [array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>weights</b>
</td>
<td>
a list of NumPy arrays. The number
of arrays and their shape must match
number of the dimensions of the weights
of the layer (i.e. it should match the
output of <b>get_weights</b>).
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
If the provided weights list does not match the
layer's specifications.
</td>
</tr>
</table>



<h3 id="__call__">__call__</h3>

```python
__call__(
    *args, **kwargs
)
```


Wraps <b>call</b>, applying pre- and post-processing steps.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>*args</b>
</td>
<td>
Positional arguments to be passed to <b>self.call</b>.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Keyword arguments to be passed to <b>self.call</b>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Output tensor(s).
</td>
</tr>

</table>



#### Note:

- The following optional keyword arguments are reserved for specific uses:
  * <b>training</b>: Boolean scalar tensor of Python boolean indicating
    whether the <b>call</b> is meant for training or inference.
  * <b>mask</b>: Boolean input mask.
- If the layer's <b>call</b> method takes a <b>mask</b> argument (as some Keras
  layers do), its default value will be set to the mask generated
  for <b>inputs</b> by the previous layer (if <b>input</b> did come from
  a layer that generated a corresponding mask, i.e. if it came from
  a Keras layer with masking support.
- If the layer is not built, the method will call <b>build</b>.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<b>ValueError</b>
</td>
<td>
if the layer's <b>call</b> method returns None (an invalid value).
</td>
</tr><tr>
<td>
<b>RuntimeError</b>
</td>
<td>
if <b>super().__init__()</b> was not called in the constructor.
</td>
</tr>
</table>





