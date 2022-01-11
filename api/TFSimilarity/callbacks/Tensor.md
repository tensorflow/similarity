# # # tf.Tensor([4 2 3], shape=(3,), dtype=int64)
```

Note: this is an implementation detail that is subject to change and users
should not rely on this behaviour.

For more on Tensors, see the [guide](https://tensorflow.org/guide/tensor).


`op`
An `Operation`. `Operation` that computes this tensor.
`value_index`
An `int`. Index of the operation's endpoint that produces
this tensor.
`dtype`
A `DType`. Type of elements stored in this tensor.




`TypeError`
If the op is not an `Operation`.






`device`
The name of the device on which this tensor will be produced, or None.
`dtype`
The `DType` of elements in this tensor.
`graph`
The `Graph` that contains this tensor.
`name`
The string name of this tensor.
`op`
The `Operation` that produces this tensor as an output.
`shape`
Returns a `tf.TensorShape` that represents the shape of this tensor.

```
>>> t = tf.constant([1,2,3,4,5])
>>> t.shape
TensorShape([5])
```

`tf.Tensor.shape` is equivalent to `tf.Tensor.get_shape()`.

In a `tf.function` or when building a model using
`tf.keras.Input`, they return the build-time shape of the
tensor, which may be partially unknown.

A `tf.TensorShape` is not a tensor. Use `tf.shape(t)` to get a tensor
containing the shape, calculated at runtime.

See `tf.Tensor.get_shape()`, and `tf.TensorShape` for details and examples.
`value_index`
The index of this tensor in the outputs of its `Operation`.



## Methods


```python
consumers()
```


Returns a list of `Operation`s that consume this tensor.


A list of `Operation`s.





```python
eval(
    feed_dict=None, session=None
)
```


Evaluates this tensor in a `Session`.

Note: If you are not using `compat.v1` libraries, you should not need this,
(or `feed_dict` or `Session`).  In eager execution (or within `tf.function`)
you do not need to call `eval`.

Calling this method will execute all preceding operations that
produce the inputs needed for the operation that produces this
tensor.

launched in a session, and either a default session must be
available, or `session` must be specified explicitly.


`feed_dict`
A dictionary that maps `Tensor` objects to feed values. See
`tf.Session.run` for a description of the valid feed values.
`session`
(Optional.) The `Session` to be used to evaluate this tensor. If
none, the default session will be used.



A numpy array corresponding to the value of this tensor.





```python
experimental_ref()
```


DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use ref() instead.


```python
get_shape()
```


Returns a `tf.TensorShape` that represents the shape of this tensor.

In eager execution the shape is always fully-known.

```
>>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
>>> print(a.shape)
(2, 3)
```

`tf.Tensor.get_shape()` is equivalent to `tf.Tensor.shape`.


When executing in a `tf.function` or building a model using
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
...   print("Result shape: ", a.shape)
...   return a
```

```
>>> cf = my_fun.get_concrete_function(
...   tf.TensorSpec([None, None]))
Result shape: (5, 5)
```

A `tf.TensorShape` representing the shape of this tensor.





```python
ref()
```


Returns a hashable reference object to this Tensor.

The primary use case for this API is to put tensors in a set/dictionary.
We can't put tensors in a set/dictionary as `tensor.__hash__()` is no longer
available starting Tensorflow 2.0.

The following will raise an exception starting 2.0

```
>>> x = tf.constant(5)
>>> y = tf.constant(10)
>>> z = tf.constant(10)
>>> tensor_set = {x, y, z}
Traceback (most recent call last):
  ...
TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
>>> tensor_dict = {x: 'five', y: 'ten'}
Traceback (most recent call last):
  ...
TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
```

Instead, we can use `tensor.ref()`.

```
>>> tensor_set = {x.ref(), y.ref(), z.ref()}
>>> x.ref() in tensor_set
True
>>> tensor_dict = {x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'}
>>> tensor_dict[y.ref()]
'ten'
```

Also, the reference object provides `.deref()` function that returns the
original Tensor.

```
>>> x = tf.constant(5)
>>> x.ref().deref()
```


```python
set_shape(
    shape
)
```


Updates the shape of this tensor.

Note: It is recommended to use `tf.ensure_shape` instead of
programming errors and can create guarantees for compiler
optimization.

With eager execution this operates as a shape assertion.
Here the shapes match:

```
>>> t = tf.constant([[1,2,3]])
>>> t.set_shape([1, 3])
```

Passing a `None` in the new shape allows any value for that axis:

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

When executing in a `tf.function`, or building a model using
...   print("Initial shape: ", result.shape)
...   result.set_shape([None, None])
...   print("Final shape: ", result.shape)
...   return result
```

Trace the function

```
>>> concrete_parse = my_parse.get_concrete_function(
...     tf.TensorSpec([], dtype=tf.string))
Final shape:  (None, None)
```

##### # The function still runs, even though it `set_shape([None,None])`
>>> t2 = concrete_parse(serialized)
>>> print(t2.shape)
(5, 5, 5)
```


`shape`
A `TensorShape` representing the shape of this tensor, a
`TensorShapeProto`, a list, a tuple, or None.




`ValueError`
If `shape` is not compatible with the current shape of
this tensor.




```python
__abs__(
    name=None
)
```


Computes the absolute value of a tensor.

Given a tensor of integer or floating-point values, this operation returns a
tensor of the same type, where each element contains the absolute value of the
corresponding element in the input.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float32` or `float64` that is the absolute value of each element in `x`. For
a complex number \\(a + bj\\), its absolute value is computed as
\\(\sqrt{a^2 + b^2}\\).

##### # complex number
>>> x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
>>> tf.abs(x)
array([[5.25594901],
       [6.60492241]])>
```


`x`
A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
`int32`, `int64`, `complex64` or `complex128`.
`name`
A name for the operation (optional).



A `Tensor` or `SparseTensor` of the same size, type and sparsity as `x`,
  with absolute values. Note, for `complex64` or `complex128` input, the
  returned `Tensor` will be of type `float32` or `float64`, respectively.

If `x` is a `SparseTensor`, returns
`SparseTensor(x.indices, tf.math.abs(x.values, ...), x.dense_shape)`





```python
__add__(
    y
)
```



  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
```

`TypeError`.





```python
__div__(
    y
)
```


Divides x / y elementwise (using Python 2 division operator semantics). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.




This function divides `x` and `y`, forcing Python 2 semantics. That is, if `x`
and `y` are both integers then the result will be an integer. This is in
contrast to Python 3, where division with `/` is always a float while division
with `//` is always an integer.


`x`
`Tensor` numerator of real numeric type.
`y`
`Tensor` denominator of real numeric type.
`name`
A name for the operation (optional).



`x / y` returns the quotient of x and y.





This function is deprecated in TF2. Prefer using the Tensor division operator,
`tf.divide`, or `tf.math.divide`, which obey the Python 3 division operator
semantics.






```python
__eq__(
    other
)
```



Compares two tensors element-wise for equality if they are
broadcast-compatible; or returns False if they are not broadcast-compatible.
(Note that this behavior differs from `tf.math.equal`, which raises an
exception if the two tensors are not broadcast-compatible.)

#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.




`self`
The left-hand side of the `==` operator.
`other`
The right-hand side of the `==` operator.



The result of the elementwise `==` operation, or `False` if the arguments
are not broadcast-compatible.





```python
__floordiv__(
    y
)
```


Divides `x / y` elementwise, rounding toward the most negative integer.

The same as `tf.compat.v1.div(x,y)` for integers, but uses
`tf.floor(tf.compat.v1.div(x,y))` for
floating point arguments so that the result is always an integer (though
possibly an integer represented as floating point).  This op is generated by
`x // y` floor division in Python 3 and in Python 2.7 with
`from __future__ import division`.

`x` and `y` must have the same type, and the result will have the same type
as well.


`x`
`Tensor` numerator of real numeric type.
`y`
`Tensor` denominator of real numeric type.
`name`
A name for the operation (optional).



`x / y` rounded down.





`TypeError`
If the inputs are complex.




```python
__ge__(
    y, name=None
)
```


Returns the truth value of (x >= y) element-wise.

*NOTE*: `math.greater_equal` supports broadcasting. More about broadcasting
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


`x`
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
`y`
A `Tensor`. Must have the same type as `x`.
`name`
A name for the operation (optional).



A `Tensor` of type `bool`.





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

- `tf.newaxis` is `None` as in NumPy.
- An implicit ellipsis is placed at the end of the `slice_spec`
- NumPy advanced indexing is currently not supported.



#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
custom composite tensors & other custom objects.

The API symbol is not intended to be called by users directly and does
appear in TensorFlow's generated documentation.




`tensor`
An ops.Tensor object.
`slice_spec`
The arguments to Tensor.__getitem__.
`var`
In the case of variable slice assignment, the Variable object to slice
(i.e. tensor is the read-only view of this variable).



The appropriate slice of "tensor", based on "slice_spec".





`ValueError`
If a slice range is negative size.
`TypeError`
If the slice indices aren't int, slice, ellipsis,
tf.newaxis or scalar int32/int64 tensors.




```python
__gt__(
    y, name=None
)
```


Returns the truth value of (x > y) element-wise.

*NOTE*: `math.greater` supports broadcasting. More about broadcasting
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


`x`
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
`y`
A `Tensor`. Must have the same type as `x`.
`name`
A name for the operation (optional).



A `Tensor` of type `bool`.





```python
__invert__(
    name=None
)
```






```python
__iter__()
```






```python
__le__(
    y, name=None
)
```



*NOTE*: `math.less_equal` supports broadcasting. More about broadcasting
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


`x`
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
`y`
A `Tensor`. Must have the same type as `x`.
`name`
A name for the operation (optional).



A `Tensor` of type `bool`.





```python
__len__()
```






```python
__lt__(
    y, name=None
)
```



*NOTE*: `math.less` supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

##### # # # # # `a` * `b`
array([[[ 94, 100],
        [229, 244]],
       [[508, 532],
        [697, 730]]], dtype=int32)>
```

Since python >= 3.5 the @ operator is supported
(see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
it simply calls the `tf.matmul()` function, so the following lines are
equivalent:

```
>>> d = a @ b @ [[10], [11]]
>>> d = tf.matmul(tf.matmul(a, b), [[10], [11]])
```


`a`
`tf.Tensor` of type `float16`, `float32`, `float64`, `int32`,
`complex64`, `complex128` and rank > 1.
`b`
`tf.Tensor` with same type and rank as `a`.
`transpose_a`
If `True`, `a` is transposed before multiplication.
`transpose_b`
If `True`, `b` is transposed before multiplication.
`adjoint_a`
If `True`, `a` is conjugated and transposed before
multiplication.
`adjoint_b`
If `True`, `b` is conjugated and transposed before
multiplication.
`a_is_sparse`
If `True`, `a` is treated as a sparse matrix. Notice, this
**does not support `tf.sparse.SparseTensor`**, it just makes optimizations
that assume most values in `a` are zero.
See `tf.sparse.sparse_dense_matmul`
for some support for `tf.sparse.SparseTensor` multiplication.
`b_is_sparse`
If `True`, `b` is treated as a sparse matrix. Notice, this
**does not support `tf.sparse.SparseTensor`**, it just makes optimizations
that assume most values in `a` are zero.
See `tf.sparse.sparse_dense_matmul`
for some support for `tf.sparse.SparseTensor` multiplication.
`output_type`
The output datatype if needed. Defaults to None in which case
the output_type is the same as input type. Currently only works when input
tensors are type (u)int8 and output_type can be int32.
`name`
Name for the operation (optional).



A `tf.Tensor` of the same type as `a` and `b` where each inner-most matrix
is the product of the corresponding matrices in `a` and `b`, e.g. if all
transpose or adjoint attributes are `False`:

`output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
for all indices `i`, `j`.
`Note`
This is matrix product, not element-wise product.




`ValueError`
If `transpose_a` and `adjoint_a`, or `transpose_b` and
`adjoint_b` are both set to `True`.
`TypeError`
If output_type is specified but the types of `a`, `b` and
`output_type` is not (u)int8, (u)int8 and int32.




```python
__mod__(
    y
)
```



true, this follows Python semantics in that the result here is consistent
with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

*NOTE*: `math.floormod` supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)


`x`
A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bfloat16`, `half`, `float32`, `float64`.
`y`
A `Tensor`. Must have the same type as `x`.
`name`
A name for the operation (optional).



A `Tensor`. Has the same type as `x`.





```python
__mul__(
    y
)
```


Dispatches cwise mul for "Dense*Dense" and "Dense*Sparse".



```python
__ne__(
    other
)
```



Compares two tensors element-wise for inequality if they are
broadcast-compatible; or returns True if they are not broadcast-compatible.
(Note that this behavior differs from `tf.math.not_equal`, which raises an
exception if the two tensors are not broadcast-compatible.)

#### Purpose in the API:


This method is exposed in TensorFlow's API so that library developers
```


`x`
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
`y`
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
`name`
A name for the operation (optional).



A `Tensor`.





```python
__radd__(
    x
)
```



  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
```


`x`
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
`y`
A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
`complex64`, or `complex128`.
`name`
A name for the operation (optional).



A `Tensor`.





```python
__rsub__(
    x
)
```


Returns x - y element-wise.

*NOTE*: `tf.subtract` supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range `(-inf, inf)`.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
>>> tf.subtract(y, x)
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary `-` operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
```

When subtracting two input values of different shapes, `tf.subtract` follows the
- [general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be `1`.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
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
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```


`x`
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `uint32`, `uint64`.
`y`
A `Tensor`. Must have the same type as `x`.
`name`
A name for the operation (optional).



A `Tensor`. Has the same type as `x`.





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
`x / y` division in Python 3 and in Python 2.7 with
`from __future__ import division`.  If you want integer division that rounds
down, use `x // y` or `tf.math.floordiv`.

`x` and `y` must have the same numeric type.  If the inputs are floating
point, the output will have the same type.  If the inputs are integral, the
inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
and `int64` (matching the behavior of Numpy).


`x`
`Tensor` numerator of numeric type.
`y`
`Tensor` denominator of numeric type.
`name`
A name for the operation (optional).



`x / y` evaluated in floating point.





`TypeError`
If `x` and `y` have different dtypes.




```python
__rxor__(
    x
)
```






```python
__sub__(
    y
)
```


Returns x - y element-wise.

*NOTE*: `tf.subtract` supports broadcasting. More about broadcasting
- [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Both input and output have a range `(-inf, inf)`.

Example usages below.

Subtract operation between an array and a scalar:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = 1
>>> tf.subtract(x, y)
>>> tf.subtract(y, x)
numpy=array([ 0, -1, -2, -3, -4], dtype=int32)>
```

Note that binary `-` operator can be used instead:

```
>>> x = tf.convert_to_tensor([1, 2, 3, 4, 5])
>>> y = tf.convert_to_tensor(1)
>>> x - y
```

Subtract operation between an array and a tensor of same shape:

```
>>> x = [1, 2, 3, 4, 5]
>>> y = tf.constant([5, 4, 3, 2, 1])
>>> tf.subtract(y, x)
numpy=array([ 4,  2,  0, -2, -4], dtype=int32)>
```

**Warning**: If one of the inputs (`x` or `y`) is a tensor and the other is a
non-tensor, the non-tensor input will adopt (or get casted to) the data type
of the tensor input. This can potentially cause unwanted overflow or underflow
conversion.

For example,

```
>>> x = tf.constant([1, 2], dtype=tf.int8)
>>> y = [2**8 + 1, 2**8 + 2]
>>> tf.subtract(x, y)
```

When subtracting two input values of different shapes, `tf.subtract` follows the
- [general broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)
. The two input array shapes are compared element-wise. Starting with the
trailing dimensions, the two dimensions either have to be equal or one of them
needs to be `1`.

For example,

```
>>> x = np.ones(6).reshape(2, 3, 1)
>>> y = np.ones(6).reshape(2, 1, 3)
>>> tf.subtract(x, y)
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
array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]])>
```


`x`
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `uint32`, `uint64`.
`y`
A `Tensor`. Must have the same type as `x`.
`name`
A name for the operation (optional).



A `Tensor`. Has the same type as `x`.





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
`x / y` division in Python 3 and in Python 2.7 with
`from __future__ import division`.  If you want integer division that rounds
down, use `x // y` or `tf.math.floordiv`.

`x` and `y` must have the same numeric type.  If the inputs are floating
point, the output will have the same type.  If the inputs are integral, the
inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
and `int64` (matching the behavior of Numpy).


`x`
`Tensor` numerator of numeric type.
`y`
`Tensor` denominator of numeric type.
`name`
A name for the operation (optional).



`x / y` evaluated in floating point.





`TypeError`
If `x` and `y` have different dtypes.




```python
__xor__(
    y
)
```










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

