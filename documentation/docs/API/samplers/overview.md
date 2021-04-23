# Overview

Samplers are used to create well balanced batchs from various type of dataset.
*A well balanced batch is a batch that contains at least 2 examples for each class* in the batch.
Having well balanced batches is important for many type of similarity learning
including contrastive learning because contrastive loss need at least
two examples (and sometime more) to be able to compute distances between the embeddings.

## Modules

[MultiShotMemorySampler()](multishot_memory.md): Sampler for datatasets that fit
in memory and have multiple examples (multi-shots) for each classes.

[SingleShotMemorySampler()](single_memory.md): Sampler for datatasets that fit
in memory and have a single examples (single-shot) for each class. Rely on
a user provided augmenter to create similar looking example.

[TFRecordDatasetSampler()](tfdataset_sampler.md): Sampler for TFRecordDatasets
that efficiently samples from shards stored on disk. Useful for very large
datasets and [public datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets) stored in the TFRecordDataset format.

[Sampler](sampler.md): Abstract Sampler class.
