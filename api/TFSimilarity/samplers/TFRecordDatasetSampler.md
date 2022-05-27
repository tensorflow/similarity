# TFSimilarity.samplers.TFRecordDatasetSampler





Create a [TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) based sampler.

```python
TFSimilarity.samplers.TFRecordDatasetSampler(
    shard_path: str,
    deserialization_fn: Callable,
    example_per_class: int = 2,
    batch_size: int = 32,
    shards_per_cycle: int = None,
    compression: Optional[str] = None,
    parallelism: int = tf.data.AUTOTUNE,
    async_cycle: bool = False,
    prefetch_size: Optional[int] = None,
    shard_suffix: str = *.tfrec,
    num_repeat: int = -1
) -> tf.data.Dataset
```



<!-- Placeholder for "Used in" -->

This sampler should be used when using a TFDataset or have a large
dataset that needs to be stored on file.

**WARNING**: This samplers assume that classes examples are contigious,
at least enough that you can get <b>example_per_class</b> numbers
of them consecutively. This requirements is needed to make the
sampling efficient and makes dataset constuctionn oftentime easier as
there is no need to worry about shuffling. Somewhat contigious means
its fine to have the same class in multiples shards as long as the
examples for the same classes are contigious in that shard.

Overall the way we make the sampling process is by using the
- [tf.dataset.interleaves](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
in a non orthodox way: we use its <b>block_length</b> to control the
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
<b>shard_path</b>
</td>
<td>
Directory where the shards are stored.
</td>
</tr><tr>
<td>
<b>deserialization_fn</b>
</td>
<td>
Function used to deserialize the tfRecord and
construct a valid example.
</td>
</tr><tr>
<td>
<b>example_per_class</b>
</td>
<td>
Number of example per class in each batch.
Defaults to 2.
</td>
</tr><tr>
<td>
<b>batch_size</b>
</td>
<td>
How many examples in each batch. The number of class in
the batch will be <b>batch_size // example_per_class</b>.
Defaults to 32.
</td>
</tr><tr>
<td>
<b>shards_per_cycle</b>
</td>
<td>
How many shards to use concurrently per cycle.
Default is None which is all of them. Can cause segv if too many
shards.
</td>
</tr><tr>
<td>
<b>compression</b>
</td>
<td>
Which compression was used when creating the dataset.
<b><i>None, "ZLIB", or "GZIP"</i></b> as specified in
- [TFRecordDataset documentation](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
Defaults to None.
</td>
</tr><tr>
<td>
<b>parallelism</b>
</td>
<td>
How many parallel calls to do. If not set, will let
TensorFlow decide by using <b>tf.data.AUTOTUNE</b> (-1).
</td>
</tr><tr>
<td>
<b>async_cycle</b>
</td>
<td>
If True, create a threadpool of size `batch_size //
example_per_class` and fetch inputs from the cycle shards
asynchronously; however, in practice, the default single thread setting
is faster. We only recommend setting this to True if it is absolutely
necessary.
</td>
</tr><tr>
<td>
<b>prefetch_size</b>
</td>
<td>
How many batch to precache. Defaults to 10.
</td>
</tr><tr>
<td>
<b>shard_suffix</b>
</td>
<td>
Glog pattern used to collect the shard files list.
Defaults to "*.tfrec".
</td>
</tr><tr>
<td>
<b>num_repeat</b>
</td>
<td>
How many times to repeat the dataset. Defaults to -1 (infinite).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <b>TF.data.dataset</b> ready to be consumed by the model.
</td>
</tr>

</table>

