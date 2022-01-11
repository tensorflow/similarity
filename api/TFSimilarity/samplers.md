# Module: TFSimilarity.samplers





Data Samplers generate balanced batches for smooth training.


*A well balanced batch is a batch that contains at least 2 examples for
each class present in the batch*.

Having well balanced batches is important for many types of similarity learning
including contrastive learning because contrastive losses require at least
two examples (and sometimes more) to be able to compute distances between
the embeddings.

To address this need, TensorFlow Similarity provides data samplers for
various types of datasets that:
- Ensure that batches contain at least N examples of each class present in
the batch.
- Support restricting the batches to a subset of the classes present in
the dataset.

## Classes

- [`class MultiShotMemorySampler`](../TFSimilarity/samplers/MultiShotMemorySampler.md): Base object for fitting to a sequence of data, such as a dataset.

- [`class SingleShotMemorySampler`](../TFSimilarity/samplers/SingleShotMemorySampler.md): Base object for fitting to a sequence of data, such as a dataset.

- [`class TFDatasetMultiShotMemorySampler`](../TFSimilarity/samplers/TFDatasetMultiShotMemorySampler.md): Base object for fitting to a sequence of data, such as a dataset.

## Functions

- [`TFRecordDatasetSampler(...)`](../TFSimilarity/samplers/TFRecordDatasetSampler.md): Create a [TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) based sampler.

- [`select_examples(...)`](../TFSimilarity/samplers/select_examples.md): Randomly select at most N examples per class

