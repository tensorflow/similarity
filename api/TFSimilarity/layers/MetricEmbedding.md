# TFSimilarity.layers.MetricEmbedding





L2 Normalized `Dense` layer.

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

This layer is usually used as output layer, especially when using cosine
distance as the similarity metric.

