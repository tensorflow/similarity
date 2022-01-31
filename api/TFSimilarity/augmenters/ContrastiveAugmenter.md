# TFSimilarity.augmenters.ContrastiveAugmenter





Helper class that provides a standard way to create an ABC using

Inherits From: [`Augmenter`](../../TFSimilarity/augmenters/Augmenter.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.augmenters.ContrastiveAugmenter(
    process: Callable,
    num_cpu: Optional[int] = os.cpu_count()
)
```



<!-- Placeholder for "Used in" -->
inheritance.

## Methods

<h3 id="augment">augment</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/contrastive.py#L30-L40">View source</a>

```python
augment(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    num_views: int,
    is_warmup: bool
) -> List[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>]
```





<h3 id="__call__">__call__</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/augmenter.py#L28-L31">View source</a>

```python
__call__(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    num_augmentations_per_example: int,
    is_warmup: bool
) -> List[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>]
```


Call self as a function.




