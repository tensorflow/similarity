# TFSimilarity.architectures.EfficientNetSim





Build an EffecientNet Model backbone for similarity learning

```python
TFSimilarity.architectures.EfficientNetSim(
    input_shape: Tuple[int],
    embedding_size: int = 128,
    variant: str = B0,
    weights: str = imagenet,
    augmentation: Union[Callable, str] = basic,
    trainable: str = frozen,
    l2_norm: bool = (True)
)
```



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

        l2_norm: If True, tensorflow_similarity.layers.MetricEmbedding is used
        as the last layer, otherwise keras.layers.Dense is used. This should be
        true when using cosine distance. Defaults to True.

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

    