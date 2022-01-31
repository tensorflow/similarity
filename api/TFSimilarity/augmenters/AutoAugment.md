# TFSimilarity.augmenters.AutoAugment





Applies the AutoAugment policy to images.

```python
TFSimilarity.augmenters.AutoAugment(
    augmentation_name: Text = v0,
    policies: Optional[Iterable[Iterable[Tuple[Text, float, float]]]] = None,
    cutout_const: float = 100,
    translate_const: float = 250
)
```



<!-- Placeholder for "Used in" -->

AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>augmentation_name</b>
</td>
<td>
The name of the AutoAugment policy to use. The
available options are <b>v0</b>, <b>test</b>, <b>reduced_cifar10</b>, <b>svhn</b> and
<b>reduced_imagenet</b>. <b>v0</b> is the policy used for all
of the results in the paper and was found to achieve the best results on
the COCO dataset. <b>v1</b>, <b>v2</b> and <b>v3</b> are additional good policies found
on the COCO dataset that have slight variation in what operations were
used during the search procedure along with how many operations are
applied in parallel to a single image (2 vs 3). Make sure to set
<b>policies</b> to <b>None</b> (the default) if you want to set options using
<b>augmentation_name</b>.
</td>
</tr><tr>
<td>
<b>policies</b>
</td>
<td>
list of lists of tuples in the form <b>(func, prob, level)</b>,
<b>func</b> is a string name of the augmentation function, <b>prob</b> is the
probability of applying the <b>func</b> operation, <b>level</b> (or magnitude) is
the input argument for <b>func</b>. For example:
```
- [[('Equalize', 0.9, 3), ('Color', 0.7, 8)],
 [('Invert', 0.6, 5), ('Rotate', 0.2, 9), ('ShearX', 0.1, 2)], ...]
```
The outer-most list must be 3-d. The number of operations in a
sub-policy can vary from one sub-policy to another.
If you provide <b>policies</b> as input, any option set with
<b>augmentation_name</b> will get overriden as they are mutually exclusive.
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
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if <b>augmentation_name</b> is unsupported.
</td>
</tr>

</table>



## Methods

<h3 id="distort">distort</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L997-L1058">View source</a>

```python
distort(
    image: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
) -> <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>
```


Applies the AutoAugment policy to <b>image</b>.

AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

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
A version of image that now has data augmentation applied to it
based on the <b>policies</b> pass into the function.
</td>
</tr>

</table>



<h3 id="policy_reduced_cifar10">policy_reduced_cifar10</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1101-L1141">View source</a>

``<b>python
@staticmethod</b>``

```python
policy_reduced_cifar10()
```


Autoaugment policy for reduced CIFAR-10 dataset.

Result is from the AutoAugment paper: https://arxiv.org/abs/1805.09501.

Each tuple is an augmentation operation of the form
(operation, probability, magnitude). Each element in policy is a
sub-policy that will be applied sequentially on the image.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the policy.
</td>
</tr>

</table>



<h3 id="policy_reduced_imagenet">policy_reduced_imagenet</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1185-L1225">View source</a>

``<b>python
@staticmethod</b>``

```python
policy_reduced_imagenet()
```


Autoaugment policy for reduced ImageNet dataset.

Result is from the AutoAugment paper: https://arxiv.org/abs/1805.09501.

Each tuple is an augmentation operation of the form
(operation, probability, magnitude). Each element in policy is a
sub-policy that will be applied sequentially on the image.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the policy.
</td>
</tr>

</table>



<h3 id="policy_simple">policy_simple</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1227-L1246">View source</a>

``<b>python
@staticmethod</b>``

```python
policy_simple()
```


Same as <b>policy_v0</b>, except with custom ops removed.


<h3 id="policy_svhn">policy_svhn</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1143-L1183">View source</a>

``<b>python
@staticmethod</b>``

```python
policy_svhn()
```


Autoaugment policy for SVHN dataset.

Result is from the AutoAugment paper: https://arxiv.org/abs/1805.09501.

Each tuple is an augmentation operation of the form
(operation, probability, magnitude). Each element in policy is a
sub-policy that will be applied sequentially on the image.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the policy.
</td>
</tr>

</table>



<h3 id="policy_test">policy_test</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1248-L1254">View source</a>

``<b>python
@staticmethod</b>``

```python
policy_test()
```


Autoaugment test policy for debugging.


<h3 id="policy_v0">policy_v0</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/augmenters/img_augments.py#L1060-L1099">View source</a>

``<b>python
@staticmethod</b>``

```python
policy_v0()
```


Autoaugment policy that was used in AutoAugment Paper.

Each tuple is an augmentation operation of the form
(operation, probability, magnitude). Each element in policy is a
sub-policy that will be applied sequentially on the image.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
the policy.
</td>
</tr>

</table>





