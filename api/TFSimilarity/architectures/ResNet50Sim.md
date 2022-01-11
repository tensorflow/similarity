# TFSimilarity.architectures.ResNet50Sim





Build an ResNet50 Model backbone for similarity learning

```python
TFSimilarity.architectures.ResNet50Sim(
    input_shape: Tuple[int],
    embedding_size: int = 128,
    weights: str = imagenet,
    trainable: str = frozen,
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = gem,
    gem_p=1.0
```



<!-- Placeholder for "Used in" -->

Architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>input_shape</b>
</td>
<td>
Size of the image input prior to augmentation,
must be bigger than the size of ResNet version you use. See below for
min input size of 244.
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
Make the ResNet backbone fully trainable or partially
trainable.
- "full" to make the entire backbone trainable,
- "partial" to only make the last conv5_block trainable
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
of 1.0 is equivelent to GlobalMeanPooling2D, while larger values
will increase the contrast between activations within each feature
map, and a value of math.inf will be equivelent to MaxPool2d.
</td>
</tr>
</table>

