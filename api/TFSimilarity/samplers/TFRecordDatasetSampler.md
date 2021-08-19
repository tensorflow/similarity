# TFSimilarity.samplers.TFRecordDatasetSampler
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/tfrecords_samplers.py#L6-L110">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Create a [TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.samplers.TFRecordDatasetSampler(
    shard_path: str,
    deserialization_fn: Callable,
    example_per_class: int = 2,
    batch_size: int = 32,
    shards_per_cycle: int = None,
    compression: Optional[str] = None,
    parallelism: int = tf.data.AUTOTUNE,
    file_parallelism: int = 1,
    prefetch_size: Optional[int] = None,
    shard_suffix: str = &#x27;*.tfrec&#x27;
) -> tf.data.Dataset
</code></pre>

<!-- Placeholder for "Used in" -->
based sampler
This sampler should be used when using a TFDataset or have a large
dataset that needs to be stored on file.
**WARNING**: This samplers assume that classes examples are contigious,
at least enough that you can get `example_per_class` numbers
of them consecutively. This requirements is needed to make the
sampling efficient and makes dataset constuctionn oftentime easier as
there is no need to worry about shuffling. Somewhat contigious means
its fine to have the same class in multiples shards as long as the
examples for the same classes are contigious in that shard.
Overall the way we make the sampling process is by using the
- [tf.dataset.interleaves](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
in a non orthodox way: we use its `block_length` to control the
number of example per class and rely on the parallelize &
non_deterministic version of the API to do the sampling efficiently
for us. Relying on pure tf.data ops also ensure good compatibility with
distribution strategy.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`shard_path`
</td>
<td>
Directory where the shards are stored.
</td>
</tr><tr>
<td>
`deserialization_fn`
</td>
<td>
Function used to deserialize the tfRecord and
construct a valid example.
</td>
</tr><tr>
<td>
`example_per_class`
</td>
<td>
Number of example per class in each batch.
Defaults to 2.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
How many examples in each batch. The number of class in
the batch will be `batch_size // example_per_class`.
Defaults to 32.
</td>
</tr><tr>
<td>
`shards_per_cycle`
</td>
<td>
How many shards to use concurrently per cycle.
Default is None which is all of them. Can cause segv if too many shards.
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
Which compression was used when creating the dataset. `{None, "ZLIB", or "GZIP"}` as specified in [TFRecordDataset documentation](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
Defaults to None.
</td>
</tr><tr>
<td>
`parallelism`
</td>
<td>
How many parallel calls to do. If not set, will let
TensorFlow decide by using `tf.data.AUTOTUNE` (-1).
</td>
</tr><tr>
<td>
`file_parallelism`
</td>
<td>
How many parallel shards to read increase number
if IO bound. Defaults to 1.
</td>
</tr><tr>
<td>
`prefetch_size`
</td>
<td>
How many batch to precache. Defaults to 10.
</td>
</tr><tr>
<td>
`shard_suffix`
</td>
<td>
Glog pattern used to collect the shard files list.
Defaults to "*.tfrec".
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `TF.data.dataset` ready to be consumed by the model.
</td>
</tr>
</table>
