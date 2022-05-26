# TFSimilarity.layers.GeneralizedMeanPooling2D





Computes the Generalized Mean of each channel in a tensor.

Inherits From: [`GeneralizedMeanPooling`](../../TFSimilarity/layers/GeneralizedMeanPooling.md)

```python
TFSimilarity.layers.GeneralizedMeanPooling2D(
    p: float = 3.0,
    data_format: Optional[str] = None,
    keepdims: bool = False,
    **kwargs
) -> None
```



<!-- Placeholder for "Used in" -->

$$
\textbf<i>e} = \left[\left(\frac{1}{|\Omega|}\sum_{u\in{\Omega}}x^{p}_{cu}\right)^{\frac{1}{p}}\right]_{c=1,\cdots,C</i>
$$

The Generalized Mean (GeM) provides a parameter <b>p</b> that sets an exponent
enabling the pooling to increase or decrease the contrast between salient
features in the feature map.

The pooling is equal to GlobalAveragePooling2D when <b>p</b> is 1.0 and equal
to MaxPool2D when <b>p</b> is <b>inf</b>.

This implementation shifts the feature map values such that the minimum
value is equal to 1.0, then computes the mean pooling, and finally shifts
the values back. This ensures that all values are positive as the
generalized mean is only valid over positive real values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>p</b>
</td>
<td>
Set the power of the mean. A value of 1.0 is equivalent to the
arithmetic mean, while a value of <b>inf</b> is equivalent to MaxPool2D.
Note, math.inf, -math.inf, and 0.0 are all supported, as well as most
positive and negative values of <b>p</b>. However, large positive values for
<b>p</b> may lead to overflow. In practice, math.inf should be used for any
<b>p</b> larger than > 25.
</td>
</tr><tr>
<td>
<b>data_format</b>
</td>
<td>
One of <b>channels_last</b> (default) or <b>channels_first</b>. The
ordering of the dimensions in the inputs.  <b>channels_last</b>
corresponds to inputs with shape <b>(batch, steps, features)</b> while
<b>channels_first</b> corresponds to inputs with shape
<b>(batch, features, steps)</b>.
</td>
</tr><tr>
<td>
<b>keepdims</b>
</td>
<td>
A boolean, whether to keep the temporal dimension or not.
If <b>keepdims</b> is <b>False</b> (default), the rank of the tensor is reduced
for spatial dimensions.  If <b>keepdims</b> is <b>True</b>, the temporal
dimension are retained with length 1.  The behavior is the same as
for <b>tf.reduce_max</b> or <b>np.max</b>.
</td>
</tr>
</table>



#### Input shape:

- If <b>data_format='channels_last'</b>:
  3D tensor with shape:
  <b>(batch_size, steps, features)</b>
- If <b>data_format='channels_first'</b>:
  3D tensor with shape:
  <b>(batch_size, features, steps)</b>


#### Output shape:

- If <b>keepdims</b>=False:
  2D tensor with shape <b>(batch_size, features)</b>.
- If <b>keepdims</b>=True:
  - If <b>data_format='channels_last'</b>:
    3D tensor with shape <b>(batch_size, 1, features)</b>
  - If <b>data_format='channels_first'</b>:
    3D tensor with shape <b>(batch_size, features, 1)</b>


