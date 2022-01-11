# TFSimilarity.layers.ActivationStdLoggingLayer





Computes the mean std of the activations of a layer.

```python
TFSimilarity.layers.ActivationStdLoggingLayer(
    name, **kwargs
)
```



<!-- Placeholder for "Used in" -->

x = reduce_std(l2_normalize(inputs, axis=0), axis=-1)

And then aggregate the per-batch mean of x over each epoch.

