# TFSimilarity.augmenters.MixupAndCutmix





Applies Mixup and/or Cutmix to a batch of images.

```python
TFSimilarity.augmenters.MixupAndCutmix(
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    prob: float = 1.0,
    switch_prob: float = 0.5,
    label_smoothing: float = 0.1,
    num_classes: int = 1001
)
```



<!-- Placeholder for "Used in" -->

- Mixup: https://arxiv.org/abs/1710.09412
- Cutmix: https://arxiv.org/abs/1905.04899

Implementaion is inspired by https://github.com/rwightman/pytorch-image-models

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
mixup_alpha (float, optional): For drawing a random lambda (<b>lam</b>) from a
  beta distribution (for each image). If zero Mixup is deactivated.
  Defaults to .8.
cutmix_alpha (float, optional): For drawing a random lambda (<b>lam</b>) from a
  beta distribution (for each image). If zero Cutmix is deactivated.
  Defaults to 1..
prob (float, optional): Of augmenting the batch. Defaults to 1.0.
switch_prob (float, optional): Probability of applying Cutmix for the
  batch. Defaults to 0.5.
label_smoothing (float, optional): Constant for label smoothing. Defaults
  to 0.1.
num_classes (int, optional): Number of classes. Defaults to 1001.
</td>
</tr>

</table>



## Methods

<h3 id="distort">distort</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1576-L1610">View source</a>

```python
distort(
    images: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    labels: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor``<b>
</a>, <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor</b>``
</a>]
```


Applies Mixup and/or Cutmix to batch of images and transforms labels.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
images (tf.Tensor): Of shape [batch_size,height, width, 3] representing a
  batch of image.
labels (tf.Tensor): Of shape [batch_size, ] representing the class id for
  each image of the batch.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Tuple[tf.Tensor, tf.Tensor]: The augmented version of <b>image</b> and
<b>labels</b>.
</td>
</tr>

</table>



<h3 id="__call__">__call__</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1571-L1574">View source</a>

```python
__call__(
    images: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    labels: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor``<b>
</a>, <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor</b>``
</a>]
```


Call self as a function.




