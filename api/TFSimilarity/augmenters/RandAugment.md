# TFSimilarity.augmenters.RandAugment





Applies the RandAugment policy to images.

```python
TFSimilarity.augmenters.RandAugment(
    num_layers: int = 2,
    magnitude: float = 10.0,
    cutout_const: float = 40.0,
    translate_const: float = 100.0,
    magnitude_std: float = 0.0,
    prob_to_apply: Optional[float] = None,
    exclude_ops: Optional[List[str]] = None
)
```



<!-- Placeholder for "Used in" -->

RandAugment is from the paper https://arxiv.org/abs/1909.13719,

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>num_layers</b>
</td>
<td>
Integer, the number of augmentation transformations to apply
sequentially to an image. Represented as (N) in the paper. Usually best
values will be in the range [1, 3].
</td>
</tr><tr>
<td>
<b>magnitude</b>
</td>
<td>
Integer, shared magnitude across all augmentation operations.
Represented as (M) in the paper. Usually best values are in the range
- [5, 10].
</td>
</tr><tr>
<td>
<b>cutout_const</b>
</td>
<td>
multiplier for applying cutout.
</td>
</tr><tr>
<td>
<b>translate_const</b>
</td>
<td>
multiplier for applying translation.
</td>
</tr><tr>
<td>
<b>magnitude_std</b>
</td>
<td>
randomness of the severity as proposed by the authors of
the timm library.
</td>
</tr><tr>
<td>
<b>prob_to_apply</b>
</td>
<td>
The probability to apply the selected augmentation at each
layer.
</td>
</tr><tr>
<td>
<b>exclude_ops</b>
</td>
<td>
exclude selected operations.
</td>
</tr>
</table>



## Methods

<h3 id="distort">distort</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1321-L1386">View source</a>

```python
distort(
    image: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
```


Applies the RandAugment policy to <b>image</b>.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>image</b>
</td>
<td>
<b>Tensor</b> of shape [height, width, 3] representing an image.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The augmented version of <b>image</b>.
</td>
</tr>

</table>





