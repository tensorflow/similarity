# TFSimilarity.losses.SimCLRLoss


used in [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)
code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)


`reduction`
Type of `tf.keras.losses.Reduction` to apply to
loss. Default value is `AUTO`. `AUTO` indicates that the reduction
option will be determined by the usage context. For almost all cases
this defaults to `SUM_OVER_BATCH_SIZE`. When used with
`tf.distribute.Strategy`, outside of built-in training loops such as
`tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
will raise an error. Please see this custom training [tutorial](
  https://www.tensorflow.org/tutorials/distribute/custom_training) for
    more details.
`name`
Optional name for the instance.



## Methods


```python
@classmethod```

```python
from_config(
    config
)
```


Instantiates a `Loss` from its config (output of `get_config()`).



`config`
Output of `get_config()`.



A `Loss` instance.






```python
get_config() -> Dict[str, Any]
```


Returns the config dictionary for a `Loss` instance.



```python
__call__(
    y_true, y_pred, sample_weight=None
)
```


Invokes the `Loss` instance.



`y_true`
Ground truth values. shape = `[batch_size, d0, .. dN]`, except
sparse loss functions such as sparse categorical crossentropy where
shape = `[batch_size, d0, .. dN-1]`
`y_pred`
The predicted values. shape = `[batch_size, d0, .. dN]`
`sample_weight`
Optional `sample_weight` acts as a coefficient for the
loss. If a scalar is provided, then the loss is simply scaled by the
given value. If `sample_weight` is a tensor of size `[batch_size]`, then
the total loss for each sample of the batch is rescaled by the
corresponding element in the `sample_weight` vector. If the shape of
`sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
this shape), then each loss element of `y_pred` is scaled
by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
  functions reduce by 1 dimension, usually axis=-1.)



Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
because all loss functions reduce by 1 dimension, usually axis=-1.)





`ValueError`
If the shape of `sample_weight` is invalid.





