# TFSimilarity.augmenters.SimCLRAugmenter





SimCLR augmentation pipeline as defined in

Inherits From: [`Augmenter`](../../TFSimilarity/augmenters/Augmenter.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.augmenters.SimCLRAugmenter(
    height: int,
    width: int,
    is_training: bool = True,
    color_distort: bool = True,
    jitter_stength: float = 1.0,
    crop: bool = True,
    eval_crop_proportion: float = 0.875,
    flip: bool = True,
    version: str = v2,
    num_cpu: Optional[int] = os.cpu_count()
)
```



<!-- Placeholder for "Used in" -->
- [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)

code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)

## Methods

<h3 id="augment">augment</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/simclr.py#L648-L664">View source</a>

``<b>python
@tf.function</b>``

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




