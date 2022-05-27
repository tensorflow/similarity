# TFSimilarity.augmenters.BarlowAugmenter





Helper class that provides a standard way to create an ABC using

Inherits From: [`Augmenter`](../../TFSimilarity/augmenters/Augmenter.md), [`ABC`](../../TFSimilarity/distances/ABC.md)

```python
TFSimilarity.augmenters.BarlowAugmenter(
    width: int,
    height: int,
    flip_probability=0.5,
    brightness_multiplier=0.8,
    contrast_multiplier=0.6,
    saturation_multiplier=0.6,
    hue_multiplier=0.2,
    jitter_probability=0.8,
    greyscale_probability=0.2,
    blur_probability=0.2,
    blur_min_sigma=0,
    blur_max_sigma=1,
    solarize_probability=0.2,
    solarize_pixel_min=0,
    solarize_pixel_max=255,
    solarize_thresh=10,
    num_cpu: Optional[int] = os.cpu_count()
)
```



<!-- Placeholder for "Used in" -->
inheritance.

## Methods

<h3 id="augment">augment</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/augmenters/barlow.py#L118-L161">View source</a>

``<b>python
@tf.function</b>``

```python
augment(
    x: Any,
    y: Any = tf.constant([0]),
    num_augmentations_per_example: int = 2,
    is_warmup: bool = True
) -> List[Any]
```





<h3 id="__call__">__call__</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/augmenters/barlow.py#L163-L170">View source</a>

```python
__call__(
    x: Any,
    y: Any = tf.constant([0]),
    num_augmentations_per_example: int = 2,
    is_warmup: bool = True
) -> List[Any]
```


Call self as a function.




