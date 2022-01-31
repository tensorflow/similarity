# TFSimilarity.augmenters.RandomErasing





Applies RandomErasing to a single image.

```python
TFSimilarity.augmenters.RandomErasing(
    probability: float = 0.25,
    min_area: float = 0.02,
    max_area: float = (1 / 3),
    min_aspect: float = 0.3,
    max_aspect=None,
    min_count=1,
    max_count=1,
    trials=10
)
```



<!-- Placeholder for "Used in" -->

Reference: https://arxiv.org/abs/1708.04896

Implementaion is inspired by https://github.com/rwightman/pytorch-image-models

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
probability (float, optional): Probability of augmenting the image.
  Defaults to 0.25.
min_area (float, optional): Minimum area of the random erasing rectangle.
  Defaults to 0.02.
max_area (float, optional): Maximum area of the random erasing rectangle.
  Defaults to 1/3.
min_aspect (float, optional): Minimum aspect rate of the random erasing
  rectangle. Defaults to 0.3.
max_aspect ([type], optional): Maximum aspect rate of the random erasing
  rectangle. Defaults to None.
min_count (int, optional): Minimum number of erased rectangles. Defaults
  to 1.
max_count (int, optional):  Maximum number of erased rectangles. Defaults
  to 1.
trials (int, optional): Maximum number of trials to randomly sample a
  rectangle that fulfills constraint. Defaults to 10.
</td>
</tr>

</table>



## Methods

<h3 id="distort">distort</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1437-L1449">View source</a>

```python
distort(
    image: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
```


Applies RandomErasing to single <b>image</b>.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
image (tf.Tensor): Of shape [height, width, 3] representing an image.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
<b>tf.Tensor</b>
</td>
<td>
The augmented version of <b>image</b>.
</td>
</tr>
</table>





