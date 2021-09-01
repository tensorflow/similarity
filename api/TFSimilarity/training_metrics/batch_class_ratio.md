# TFSimilarity.training_metrics.batch_class_ratio





Computes the average number of examples per class within each batch.

```python
TFSimilarity.training_metrics.batch_class_ratio(
    sampler: Sampler,
    num_batches: int = 100
) -> float
```



<!-- Placeholder for "Used in" -->
Similarity learning requires at least 2 examples per class in each batch.
This is needed in order to construct the triplets. This function
provides the average number of examples per class within each batch and
can be used to check that a sampler is working correctly.
The ratio should be >= 2.
Args:
    sampler: A tf.similarity sampler object.
    num_batches: The number of batches to sample.
Returns:
    The average number of examples per class.