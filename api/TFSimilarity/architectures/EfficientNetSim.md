
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.architectures.EfficientNetSim" />
<meta itemprop="path" content="Stable" />
</div>
# TFSimilarity.architectures.EfficientNetSim
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/architectures/efficientnet.py#L33-L103">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Build an EffecientNet Model backbone for similarity learning
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.architectures.EfficientNetSim(
    input_shape: Tuple[int],
    embedding_size: int = 128,
    variant: str = &#x27;B0&#x27;,
    weights: str = &#x27;imagenet&#x27;,
    augmentation: Union[Callable, str] = &#x27;basic&#x27;,
    trainable: str = &#x27;frozen&#x27;
)
</code></pre>

<!-- Placeholder for "Used in" -->
    Architecture from [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
](https://arxiv.org/abs/1905.11946)
    Args:
        input_shape: Size of the image input prior to augmentation,
        must be bigger than the size of Effnet version you use. See below for
        min input size.
        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.
        variant: Which Variant of the EfficientNEt to use. Defaults to "B0".
        weights: Use pre-trained weights - the only available currently being
        imagenet. Defaults to "imagenet".
        augmentation: How to augment the data - either pass a Sequential model
        of keras.preprocessing.layers or use the built in one or set it to
        None to disable. Defaults to "basic".
        trainable: Make the EfficienNet backbone fully trainable or partially
        trainable. Either "full" to make the entire backbone trainable,
        "partial" to only make the last 3 block trainable or "frozen" to make
        it not trainable. Defaults to "frozen".
    Note:
        EfficientNet expects images at the following size:
         - "B0": 224,
         - "B1": 240,
         - "B2": 260,
         - "B3": 300,
         - "B4": 380,
         - "B5": 456,
         - "B6": 528,
         - "B7": 600,
    