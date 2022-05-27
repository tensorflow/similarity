# TFSimilarity.utils.BoolTensor





Bool tensor 

Inherits From: [`Tensor`](../../TFSimilarity/callbacks/Tensor.md)

```python
TFSimilarity.utils.BoolTensor(
    op, value_index, dtype
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>op</b>
</td>
<td>
An <b>Operation</b>. <b>Operation</b> that computes this tensor.
</td>
</tr><tr>
<td>
<b>value_index</b>
</td>
<td>
An <b>int</b>. Index of the operation's endpoint that produces
this tensor.
</td>
</tr><tr>
<td>
<b>dtype</b>
</td>
<td>
A <b>DType</b>. Type of elements stored in this tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<b>TypeError</b>
</td>
<td>
If the op is not an <b>Operation</b>.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>device</b>
</td>
<td>
The name of the device on which this tensor will be produced, or None.
</td>
</tr><tr>
<td>
<b>dtype</b>
</td>
<td>
The <b>DType</b> of elements in this tensor.
</td>
</tr><tr>
<td>
<b>graph</b>
</td>
<td>
The <b>Graph</b> that contains this tensor.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
The string name of this tensor.
</td>
</tr><tr>
<td>
<b>op</b>
</td>
<td>
The <b>Operation</b> that produces this tensor as an output.
</td>
</tr><tr>
<td>
<b>shape</b>
</td>
<td>
Returns a <b>tf.TensorShape</b> that represents the shape of this tensor.

```
>>> t = tf.constant([1,2,3,4,5])
>>> t.shape
TensorShape([5])
```

<b>tf.Tensor.shape</b> is equivalent to <b>tf.Tensor.get_shape()</b>.

In a <b>tf.function</b> or when building a model using
<b>tf.keras.Input</b>, they return the build-time shape of the
tensor, which may be partially unknown.

A <b>tf.TensorShape</b> is not a tensor. Use <b>tf.shape(t)</b> to get a tensor
containing the shape, calculated at runtime.

See <b>tf.Tensor.get_shape()</b>, and <b>tf.TensorShape</b> for details and examples.
</td>
</tr><tr>
<td>
<b>value_index</b>
</td>
<td>
The index of this tensor in the outputs of its <b>Operation</b>.
</td>
</tr>
</table>



## Methods

<h3 id="consumers">consumers</h3>

```python
consumers()
```


Returns a list of <b>Operation</b>s that consume this tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of <b>Operation</b>s.
</td>
</tr>

</table>



<h3 id="eval">eval</h3>

```python
eval(
    feed_dict=None, session=None
)
```


Evaluates this tensor in a <b>Session</b>.

Note: If you are not using <b>compat.v1</b> libraries, you should not need this,
(or <b>feed_dict</b> or <b>Session</b>).  In eager execution (or within <b>tf.function</b>)
you do not need to call <b>eval</b>.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

*N.B.* Before invoking <a href="../../TFSimilarity/callbacks/Tensor.md#eval">Tensor.eval()```
</a>, its graph must have been
launched in a session, and either a default session must be
available, or <b>session</b> must be specified explicitly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>feed_dict</b>
</td>
<td>
A dictionary that maps <b>Tensor</b> objects to feed values. See
<b>tf.Session.run</b> for a description of the valid feed values.
</td>
</tr><tr>
<td>
<b>session</b>
</td>
<td>
(Optional.) The <b>Session</b> to be used to evaluate this tensor. If
none, the default session will be used.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A numpy array corresponding to the value of this tensor.
</td>
</tr>

</table>



<h3 id="experimental_ref">experimental_ref</h3>

```python
experimental_ref()
```


DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use ref() instead.

<h3 id="get_shape">get_shape</h3>

```python
get_shape()
```


Returns a <b>tf.TensorShape</b> that represents the shape of this tensor.

In eager execution the shape is always fully-known.

```
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> print(a.shape)
(2, 3)
```

<b>tf.Tensor.get_shape()</b> is equivalent to <b>tf.Tensor.shape</b>.


When executing in a <b>tf.function</b> or building a model using
<b>tf.keras.Input</b>, <a href="../../TFSimilarity/callbacks/Tensor.md## # the <b>print</b> executes during tracing.
...   print("Result shape: ", result.shape)
...   return result
```

The shape inference functions propagate shapes to the extent possible:

```
>>> f = my_matmul.get_concrete_function(
...   tf.TensorSpec([None,3]),
...   tf.TensorSpec([3,5]))
Result shape: (None, 5)
```

Tracing may fail if a shape missmatch can be detected:

```
>>> cf = my_matmul.get_concrete_function(
...   tf.TensorSpec([None,3]),
...   tf.TensorSpec([4,5]))
Traceback (most recent call last):
...
ValueError: Dimensions must be equal, but are 3 and 4 for 'matmul' (op:
'MatMul') with input shapes: [?,3], [4,5].
```

In some cases, the inferred shape may have unknown dimensions. If
the caller has additional information about the values of these
dimensions, <b>tf.ensure_shape</b> or <a href="../../TFSimilarity/callbacks/Tensor.md## the <b>print</b> executes during tracing.
...   print("Result shape: ", a.shape)
...   return a
```

```
>>> cf = my_fun.get_concrete_function(
...   tf.TensorSpec([None, None]))
Result shape: (5, 5)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>tf.TensorShape</b> representing the shape of this tensor.
</td>
</tr>

</table>



<h3 id="ref">ref</h3>

```python
ref()
```


Returns a hashable reference object to this Tensor.

The primary use case for this API is to put tensors in a set/dictionary.
We can't put tensors in a set/dictionary as <b>tensor.__hash__()</b> is no longer
available starting Tensorflow 2.0.

The following will raise an exception starting 2.0

```
>>> x = tf.constant(5)
>>> y = tf.constant(10)
>>> z = tf.constant(10)
>>> tensor_set = <i>x, y, z</i>
Traceback (most recent call last):
  ...
TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
>>> tensor_dict = <i>x: 'five', y: 'ten'</i>
Traceback (most recent call last):
  ...
TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
```

Instead, we can use <b>tensor.ref()</b>.

```
>>> tensor_set = <i>x.ref(), y.ref(), z.ref()</i>
>>> x.ref() in tensor_set
True
>>> tensor_dict = <i>x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'</i>
>>> tensor_dict[y.ref()]
'ten'
```

Also, the reference object provides <b>.deref()</b> function that returns the
original Tensor.

```
>>> x = tf.constant(5)
>>> x.ref().deref()
<tf.Tensor: shape=(), dtype=int32, numpy=5>
```

<h3 id="set_shape">set_shape</h3>

```python
set_shape(
    shape
)
```


Updates the shape of this tensor.

Note: It is recommended to use <b>tf.ensure_shape</b> instead of
<a href="../../TFSimilarity/callbacks/Tensor.md#set_shape">Tensor.set_shape``<b>
</a>, because </b>tf.ensure_shape` provides better checking for
programming errors and can create guarantees for compiler
optimization.

With eager execution this operates as a shape assertion.
Here the shapes match:

```
>>> t = tf.constant([[1,2,3]])
>>> t.set_shape([1, 3])
```

Passing a <b>None</b> in the new shape allows any value for that axis:

```
>>> t.set_shape([1,None])
```

An error is raised if an incompatible shape is passed.

```
>>> t.set_shape([1,5])
Traceback (most recent call last):
...
ValueError: Tensor's shape (1, 3) is not compatible with supplied
shape [1, 5]
```

When executing in a <b>tf.function</b>, or building a model using
<b>tf.keras.Input</b>, <a href="../../TFSimilarity/callbacks/Tensor.md## # the <b>print</b> executes during tracing.
...   print("Initial shape: ", result.shape)
...   result.set_shape([None, None])
...   print("Final shape: ", result.shape)
...   return result
```

Trace the function

```
>>> concrete_parse = my_parse.get_concrete_function(
...     tf.TensorSpec([], dtype=tf.string))
Initial shape:  <unknown>
Final shape:  (None, None)
```

##### # The function still runs, even though it <b>set_shape([None,None])</b>
>>> t2 = concrete_parse(serialized)
>>> print(t2.shape)
(5, 5, 5)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>shape</b>
</td>
<td>
A <b>TensorShape</b> representing the shape of this tensor, a
<b>TensorShapeProto</b>, a list, a tuple, or None.
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
If <b>shape</b> is not compatible with the current shape of
this tensor.
</td>
</tr>
</table>



<h3 id="__abs__">__abs__</h3>

```python
__abs__(
    name=None
)
```


Computes the absolute value of a tensor.

Given a tensor of integer or floating-point values, this operation returns a
tensor of the same type, where each element contains the absolute value of the
corresponding element in the input.

Given a tensor <b>x</b> of complex numbers, this operation returns a tensor of type
<b>float32</b> or <b>float64</b> that is the absolute value of each element in <b>x</b>. For
a complex number \\(a + bj\\), its absolute value is computed as
\\(\sqrt<i>a^2 + b^2</i>\\).

##### # complex number
>>> x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
>>> tf.abs(x)
<tf.Tensor: shape=(2, 1), dtype=float64, numpy=
array([[5.25594901],
       [6.60492241]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b> or <b>SparseTensor</b> of type <b>float16</b>, <b>float32</b>, <b>float64</b>,
<b>int32</b>, <b>int64</b>, <b>complex64</b> or <b>complex128</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b> or <b>SparseTensor</b> of the same size, type and sparsity as <b>x</b>,
  with absolute values. Note, for <b>complex64</b> or <b>complex128</b> input, the
  returned <b>Tensor</b> will be of type <b>float32</b> or <b>float64</b>, respectively.

If <b>x</b> is a <b>SparseTensor</b>, returns
<b>SparseTensor(x.indices, tf.math.abs(x.values, ...), x.dense_shape)</b>
</td>
</tr>

</table>



<h3 id="__add__">__add__</h3>

```python
__add__(
    y
)
```


The operation invoked by the <a href="../../TFSimilarity/callbacks/Tensor.md#__add__">Tensor.__add__```
</a> operator.


#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../../TFSimilarity/callbacks/Tensor.md## # # # ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
<b>TypeError</b>.
</td>
</tr>

</table>



<h3 id="__div__">__div__</h3>

```python
__div__(
    y
)
```


Divides x / y elementwise (using Python 2 division operator semantics). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.




This function divides <b>x</b> and <b>y</b>, forcing Python 2 semantics. That is, if <b>x</b>
and <b>y</b> are both integers then the result will be an integer. This is in
contrast to Python 3, where division with <b>/</b> is always a float while division
with <b>//</b> is always an integer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
<b>Tensor</b> numerator of real numeric type.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
<b>Tensor</b> denominator of real numeric type.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
<b>x / y</b> returns the quotient of x and y.
</td>
</tr>

</table>



 <section><devsite-expandable >
 <h4 class="showalways">Migrate to TF2</h4>

This function is deprecated in TF2. Prefer using the Tensor division operator,
<b>tf.divide</b>, or <b>tf.math.divide</b>, which obey the Python 3 division operator
semantics.


 </devsite-expandable></section>



<h3 id="__eq__">__eq__</h3>

```python
__eq__(
    other
)
```


The operation invoked by the <a href="../../TFSimilarity/callbacks/Tensor.md#__eq__">Tensor.__eq__```
</a> operator.

Compares two tensors element-wise for equality if they are
broadcast-compatible; or returns False if they are not broadcast-compatible.
(Note that this behavior differs from <b>tf.math.equal</b>, which raises an
exception if the two tensors are not broadcast-compatible.)

#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../../TFSimilarity/callbacks/Tensor.md#__eq__">Tensor.__eq__```
</a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>self</b>
</td>
<td>
The left-hand side of the <b>==</b> operator.
</td>
</tr><tr>
<td>
<b>other</b>
</td>
<td>
The right-hand side of the <b>==</b> operator.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The result of the elementwise <b>==</b> operation, or <b>False</b> if the arguments
are not broadcast-compatible.
</td>
</tr>

</table>



<h3 id="__floordiv__">__floordiv__</h3>

```python
__floordiv__(
    y
)
```


Divides <b>x / y</b> elementwise, rounding toward the most negative integer.

Mathematically, this is equivalent to floor(x / y). For example:
  floor(8.4 / 4.0) = floor(2.1) = 2.0
  floor(-8.4 / 4.0) = floor(-2.1) = -3.0
This is equivalent to the '//' operator in Python 3.0 and above.

Note: <b>x</b> and <b>y</b> must have the same type, and the result will have the same
type as well.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
<b>Tensor</b> numerator of real numeric type.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
<b>Tensor</b> denominator of real numeric type.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
<b>x / y</b> rounded toward -infinity.
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
If the inputs are complex.
</td>
</tr>
</table>



<h3 id="__ge__">__ge__</h3>

```python
__ge__(
    y, name=None
)
```


Returns the truth value of (x >= y) element-wise.

*NOTE*: <b>math.greater_equal</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6, 7])
y = tf.constant([5, 2, 5, 10])
tf.math.greater_equal(x, y) ==> [True, True, True, False]

x = tf.constant([5, 4, 6, 7])
y = tf.constant([5])
tf.math.greater_equal(x, y) ==> [True, False, True, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b>. Must be one of the following types: <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>uint8</b>, <b>int16</b>, <b>int8</b>, <b>int64</b>, <b>bfloat16</b>, <b>uint16</b>, <b>half</b>, <b>uint32</b>, <b>uint64</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b>. Must have the same type as <b>x</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b> of type <b>bool</b>.
</td>
</tr>

</table>



<h3 id="__getitem__">__getitem__</h3>

```python
__getitem__(
    slice_spec, var=None
)
```


Overload for Tensor.__getitem__.

This operation extracts the specified region from the tensor.
The notation is similar to NumPy with the restriction that
currently only support basic indexing. That means that
using a non-scalar tensor as input is not currently allowed.

##### # # # # # # # # # # # # # # # => [3, 4, 5, 6, 7, 8, 9]
```

#### Notes:

- <b>tf.newaxis</b> is <b>None</b> as in NumPy.
- An implicit ellipsis is placed at the end of the <b>slice_spec</b>
- NumPy advanced indexing is currently not supported.



#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../../TFSimilarity/callbacks/Tensor.md#__getitem__">Tensor.__getitem__```
</a> to allow it to handle
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>tensor</b>
</td>
<td>
An ops.Tensor object.
</td>
</tr><tr>
<td>
<b>slice_spec</b>
</td>
<td>
The arguments to Tensor.__getitem__.
</td>
</tr><tr>
<td>
<b>var</b>
</td>
<td>
In the case of variable slice assignment, the Variable object to slice
(i.e. tensor is the read-only view of this variable).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The appropriate slice of "tensor", based on "slice_spec".
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
If a slice range is negative size.
</td>
</tr><tr>
<td>
<b>TypeError</b>
</td>
<td>
If the slice indices aren't int, slice, ellipsis,
tf.newaxis or scalar int32/int64 tensors.
</td>
</tr>
</table>



<h3 id="__gt__">__gt__</h3>

```python
__gt__(
    y, name=None
)
```


Returns the truth value of (x > y) element-wise.

*NOTE*: <b>math.greater</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6])
y = tf.constant([5, 2, 5])
tf.math.greater(x, y) ==> [False, True, True]

x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.greater(x, y) ==> [False, False, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b>. Must be one of the following types: <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>uint8</b>, <b>int16</b>, <b>int8</b>, <b>int64</b>, <b>bfloat16</b>, <b>uint16</b>, <b>half</b>, <b>uint32</b>, <b>uint64</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b>. Must have the same type as <b>x</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b> of type <b>bool</b>.
</td>
</tr>

</table>



<h3 id="__invert__">__invert__</h3>

```python
__invert__(
    name=None
)
```





<h3 id="__iter__">__iter__</h3>

```python
__iter__()
```





<h3 id="__le__">__le__</h3>

```python
__le__(
    y, name=None
)
```


Returns the truth value of (x <= y) element-wise.

*NOTE*: <b>math.less_equal</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#### Example:



```python
x = tf.constant([5, 4, 6])
y = tf.constant([5])
tf.math.less_equal(x, y) ==> [True, True, False]

x = tf.constant([5, 4, 6])
y = tf.constant([5, 6, 6])
tf.math.less_equal(x, y) ==> [True, True, True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b>. Must be one of the following types: <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>uint8</b>, <b>int16</b>, <b>int8</b>, <b>int64</b>, <b>bfloat16</b>, <b>uint16</b>, <b>half</b>, <b>uint32</b>, <b>uint64</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b>. Must have the same type as <b>x</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b> of type <b>bool</b>.
</td>
</tr>

</table>



<h3 id="__len__">__len__</h3>

```python
__len__()
```





<h3 id="__lt__">__lt__</h3>

```python
__lt__(
    y, name=None
)
```


Returns the truth value of (x < y) element-wise.

*NOTE*: <b>math.less</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### # # # # # <b>a</b> * <b>b</b>
<tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
array([[[ 94, 100],
        [229, 244]],
       [[508, 532],
        [697, 730]]], dtype=int32)>
```

Since python >= 3.5 the @ operator is supported
(see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
it simply calls the <b>tf.matmul()</b> function, so the following lines are
equivalent:

```
>>> d = a @ b @ [[10], [11]]
>>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>a</b>
</td>
<td>
<b>tf.Tensor</b> of type <b>float16</b>, <b>float32</b>, <b>float64</b>, <b>int32</b>,
<b>complex64</b>, <b>complex128</b> and rank > 1.
</td>
</tr><tr>
<td>
<b>b</b>
</td>
<td>
<b>tf.Tensor</b> with same type and rank as <b>a</b>.
</td>
</tr><tr>
<td>
<b>transpose_a</b>
</td>
<td>
If <b>True</b>, <b>a</b> is transposed before multiplication.
</td>
</tr><tr>
<td>
<b>transpose_b</b>
</td>
<td>
If <b>True</b>, <b>b</b> is transposed before multiplication.
</td>
</tr><tr>
<td>
<b>adjoint_a</b>
</td>
<td>
If <b>True</b>, <b>a</b> is conjugated and transposed before
multiplication.
</td>
</tr><tr>
<td>
<b>adjoint_b</b>
</td>
<td>
If <b>True</b>, <b>b</b> is conjugated and transposed before
multiplication.
</td>
</tr><tr>
<td>
<b>a_is_sparse</b>
</td>
<td>
If <b>True</b>, <b>a</b> is treated as a sparse matrix. Notice, this
**does not support <b>tf.sparse.SparseTensor</b>**, it just makes optimizations
that assume most values in <b>a</b> are zero.
See <b>tf.sparse.sparse_dense_matmul</b>
for some support for <b>tf.sparse.SparseTensor</b> multiplication.
</td>
</tr><tr>
<td>
<b>b_is_sparse</b>
</td>
<td>
If <b>True</b>, <b>b</b> is treated as a sparse matrix. Notice, this
**does not support <b>tf.sparse.SparseTensor</b>**, it just makes optimizations
that assume most values in <b>a</b> are zero.
See <b>tf.sparse.sparse_dense_matmul</b>
for some support for <b>tf.sparse.SparseTensor</b> multiplication.
</td>
</tr><tr>
<td>
<b>output_type</b>
</td>
<td>
The output datatype if needed. Defaults to None in which case
the output_type is the same as input type. Currently only works when input
tensors are type (u)int8 and output_type can be int32.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
Name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>tf.Tensor</b> of the same type as <b>a</b> and <b>b</b> where each inner-most matrix
is the product of the corresponding matrices in <b>a</b> and <b>b</b>, e.g. if all
transpose or adjoint attributes are <b>False</b>:

<b>output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])</b>,
for all indices <b>i</b>, <b>j</b>.
</td>
</tr>
<tr>
<td>
<b>Note</b>
</td>
<td>
This is matrix product, not element-wise product.
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
If <b>transpose_a</b> and <b>adjoint_a</b>, or <b>transpose_b</b> and
<b>adjoint_b</b> are both set to <b>True</b>.
</td>
</tr><tr>
<td>
<b>TypeError</b>
</td>
<td>
If output_type is specified but the types of <b>a</b>, <b>b</b> and
<b>output_type</b> is not (u)int8, (u)int8 and int32.
</td>
</tr>
</table>



<h3 id="__mod__">__mod__</h3>

```python
__mod__(
    y
)
```


Returns element-wise remainder of division. When <b>x < 0</b> xor <b>y < 0</b> is

true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. <b>floor(x / y) * y + mod(x, y) = x</b>.

*NOTE*: <b>math.floormod</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b>. Must be one of the following types: <b>int8</b>, <b>int16</b>, <b>int32</b>, <b>int64</b>, <b>uint8</b>, <b>uint16</b>, <b>uint32</b>, <b>uint64</b>, <b>bfloat16</b>, <b>half</b>, <b>float32</b>, <b>float64</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b>. Must have the same type as <b>x</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b>. Has the same type as <b>x</b>.
</td>
</tr>

</table>



<h3 id="__mul__">__mul__</h3>

```python
__mul__(
    y
)
```


Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".


<h3 id="__ne__">__ne__</h3>

```python
__ne__(
    other
)
```


The operation invoked by the <a href="../../TFSimilarity/callbacks/Tensor.md#__ne__">Tensor.__ne__```
</a> operator.

Compares two tensors element-wise for inequality if they are
broadcast-compatible; or returns True if they are not broadcast-compatible.
(Note that this behavior differs from <b>tf.math.not_equal</b>, which raises an
exception if the two tensors are not broadcast-compatible.)

#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../../TFSimilarity/callbacks/Tensor.md## [[256, 65536], [9, 27]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b> of type <b>float16</b>, <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>int64</b>,
<b>complex64</b>, or <b>complex128</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b> of type <b>float16</b>, <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>int64</b>,
<b>complex64</b>, or <b>complex128</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b>.
</td>
</tr>

</table>



<h3 id="__radd__">__radd__</h3>

```python
__radd__(
    x
)
```


The operation invoked by the <a href="../../TFSimilarity/callbacks/Tensor.md#__add__">Tensor.__add__```
</a> operator.


#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
can register dispatching for <a href="../../TFSimilarity/callbacks/Tensor.md## # # # # # # [[256, 65536], [9, 27]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b> of type <b>float16</b>, <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>int64</b>,
<b>complex64</b>, or <b>complex128</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b> of type <b>float16</b>, <b>float32</b>, <b>float64</b>, <b>int32</b>, <b>int64</b>,
<b>complex64</b>, or <b>complex128</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b>.
</td>
</tr>

</table>



<h3 id="__rsub__">__rsub__</h3>

```python
__rsub__(
    x
)
```


Returns x - y element-wise.

*NOTE*: <b>tf.subtract</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range <b>(-inf, inf)</b>.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary <b>-</b> operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (<b>x</b> or <b>y</b>) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([0, 0], dtype=int8)>
```

When subtracting two input values of different shapes, <b>tf.subtract</b> follows the
- [general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be <b>1</b>.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 3), dtype=float64, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])>
```

Example with inputs of different dimensions:

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(1, 6)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 6), dtype=float64, numpy=
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b>. Must be one of the following types: <b>bfloat16</b>, <b>half</b>, <b>float32</b>, <b>float64</b>, <b>uint8</b>, <b>int8</b>, <b>uint16</b>, <b>int16</b>, <b>int32</b>, <b>int64</b>, <b>complex64</b>, <b>complex128</b>, <b>uint32</b>, <b>uint64</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b>. Must have the same type as <b>x</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b>. Has the same type as <b>x</b>.
</td>
</tr>

</table>



<h3 id="__rtruediv__">__rtruediv__</h3>

```python
__rtruediv__(
    x
)
```


Divides x / y elementwise (using Python 3 division operator semantics).

NOTE: Prefer using the Tensor operator or tf.divide which obey Python
division operator semantics.

This function forces Python 3 division operator semantics where all integer
arguments are cast to floating types first.   This op is generated by normal
<b>x / y</b> division in Python 3 and in Python 2.7 with
<b>from __future__ import division</b>.  If you want integer division that rounds
down, use <b>x // y</b> or <b>tf.math.floordiv</b>.

<b>x</b> and <b>y</b> must have the same numeric type.  If the inputs are floating
point, the output will have the same type.  If the inputs are integral, the
inputs are cast to <b>float32</b> for <b>int8</b> and <b>int16</b> and <b>float64</b> for <b>int32</b>
and <b>int64</b> (matching the behavior of Numpy).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
<b>Tensor</b> numerator of numeric type.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
<b>Tensor</b> denominator of numeric type.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
<b>x / y</b> evaluated in floating point.
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
If <b>x</b> and <b>y</b> have different dtypes.
</td>
</tr>
</table>



<h3 id="__rxor__">__rxor__</h3>

```python
__rxor__(
    x
)
```





<h3 id="__sub__">__sub__</h3>

```python
__sub__(
    y
)
```


Returns x - y element-wise.

*NOTE*: <b>tf.subtract</b> supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range <b>(-inf, inf)</b>.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary <b>-</b> operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
<tf.Tensor: shape=(5,), dtype=int32,
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (<b>x</b> or <b>y</b>) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2,), dtype=int8, numpy=array([0, 0], dtype=int8)>
```

When subtracting two input values of different shapes, <b>tf.subtract</b> follows the
- [general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be <b>1</b>.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 3), dtype=float64, numpy=
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],
       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])>
```

Example with inputs of different dimensions:

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(1, 6)
>>> tf.subtract(x, y)
<tf.Tensor: shape=(2, 3, 6), dtype=float64, numpy=
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
A <b>Tensor</b>. Must be one of the following types: <b>bfloat16</b>, <b>half</b>, <b>float32</b>, <b>float64</b>, <b>uint8</b>, <b>int8</b>, <b>uint16</b>, <b>int16</b>, <b>int32</b>, <b>int64</b>, <b>complex64</b>, <b>complex128</b>, <b>uint32</b>, <b>uint64</b>.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
A <b>Tensor</b>. Must have the same type as <b>x</b>.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <b>Tensor</b>. Has the same type as <b>x</b>.
</td>
</tr>

</table>



<h3 id="__truediv__">__truediv__</h3>

```python
__truediv__(
    y
)
```


Divides x / y elementwise (using Python 3 division operator semantics).

NOTE: Prefer using the Tensor operator or tf.divide which obey Python
division operator semantics.

This function forces Python 3 division operator semantics where all integer
arguments are cast to floating types first.   This op is generated by normal
<b>x / y</b> division in Python 3 and in Python 2.7 with
<b>from __future__ import division</b>.  If you want integer division that rounds
down, use <b>x // y</b> or <b>tf.math.floordiv</b>.

<b>x</b> and <b>y</b> must have the same numeric type.  If the inputs are floating
point, the output will have the same type.  If the inputs are integral, the
inputs are cast to <b>float32</b> for <b>int8</b> and <b>int16</b> and <b>float64</b> for <b>int32</b>
and <b>int64</b> (matching the behavior of Numpy).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
<b>Tensor</b> numerator of numeric type.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
<b>Tensor</b> denominator of numeric type.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
<b>x / y</b> evaluated in floating point.
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
If <b>x</b> and <b>y</b> have different dtypes.
</td>
</tr>
</table>



<h3 id="__xor__">__xor__</h3>

```python
__xor__(
    y
)
```









<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
OVERLOADABLE_OPERATORS<a id="OVERLOADABLE_OPERATORS"></a>
</td>
<td>
```
{
 '__abs__',
 '__add__',
 '__and__',
 '__div__',
 '__eq__',
 '__floordiv__',
 '__ge__',
 '__getitem__',
 '__gt__',
 '__invert__',
 '__le__',
 '__lt__',
 '__matmul__',
 '__mod__',
 '__mul__',
 '__ne__',
 '__neg__',
 '__or__',
 '__pow__',
 '__radd__',
 '__rand__',
 '__rdiv__',
 '__rfloordiv__',
 '__rmatmul__',
 '__rmod__',
 '__rmul__',
 '__ror__',
 '__rpow__',
 '__rsub__',
 '__rtruediv__',
 '__rxor__',
 '__sub__',
 '__truediv__',
 '__xor__'
}
```
</td>
</tr>
</table>

