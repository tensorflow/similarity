# TFSimilarity.architectures.EfficientNetSim





Build an EfficientNet Model backbone for similarity learning

```python
TFSimilarity.architectures.EfficientNetSim(
    input_shape: Tuple[int, int, int],
    embedding_size: int = 128,
    variant: str = B0,
    weights: str = imagenet,
    trainable: str = frozen,
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = gem,
    gem_p=3.0
```



<!-- Placeholder for "Used in" -->

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>input_shape</b>
</td>
<td>
Size of the input image. Must match size of EfficientNet version you use.
See below for version input size.
</td>
</tr><tr>
<td>
<b>embedding_size</b>
</td>
<td>
Size of the output embedding. Usually between 64
and 512. Defaults to 128.
</td>
</tr><tr>
<td>
<b>variant</b>
</td>
<td>
Which Variant of the EfficientNet to use. Defaults to "B0".
</td>
</tr><tr>
<td>
<b>weights</b>
</td>
<td>
Use pre-trained weights - the only available currently being
imagenet. Defaults to "imagenet".
</td>
</tr><tr>
<td>
<b>trainable</b>
</td>
<td>
Make the EfficientNet backbone fully trainable or partially
trainable.
- "full" to make the entire backbone trainable,
- "partial" to only make the last 3 block trainable
- "frozen" to make it not trainable.
</td>
</tr><tr>
<td>
<b>l2_norm</b>
</td>
<td>
If True and include_top is also True, then
tfsim.layers.MetricEmbedding is used as the last layer, otherwise
keras.layers.Dense is used. This should be true when using cosine
distance. Defaults to True.
</td>
</tr><tr>
<td>
<b>include_top</b>
</td>
<td>
Whether to include the fully-connected layer at the top
of the network. Defaults to True.
</td>
</tr><tr>
<td>
<b>pooling</b>
</td>
<td>
Optional pooling mode for feature extraction when
include_top is False. Defaults to gem.
- None means that the output of the model will be the 4D tensor
  output of the last convolutional layer.
- avg means that global average pooling will be applied to the
  output of the last convolutional layer, and thus the output of the
  model will be a 2D tensor.
- max means that global max pooling will be applied.
- gem means that global GeneralizedMeanPooling2D will be applied.
  The gem_p param sets the contrast amount on the pooling.
</td>
</tr><tr>
<td>
<b>gem_p</b>
</td>
<td>
Sets the power in the GeneralizedMeanPooling2D layer. A value
of 1.0 is equivalent to GlobalMeanPooling2D, while larger values
will increase the contrast between activations within each feature
map, and a value of math.inf will be equivalent to MaxPool2d.
</td>
</tr>
</table>



#### Note:

EfficientNet expects images at the following size:
 - "B0": 224,
 - "B1": 240,
 - "B2": 260,
 - "B3": 300,
 - "B4": 380,
 - "B5": 456,
 - "B6": 528,
 - "B7": 600,
