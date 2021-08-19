# TFSimilarity.samplers.TFDatasetMultiShotMemorySampler





Base object for fitting to a sequence of data, such as a dataset.

Inherits From: [`MultiShotMemorySampler`](../../TFSimilarity/samplers/MultiShotMemorySampler.md), [`Sampler`](../../TFSimilarity/metrics/Sampler.md)

```python
TFSimilarity.samplers.TFDatasetMultiShotMemorySampler(
    dataset_name: str,
    classes_per_batch: int,
    x_key: str = image,
    y_key: str = label,
    splits: Union[str, List[str]] = [train, test],
    examples_per_class_per_batch: int = 2,
    steps_per_epoch: int = 1000,
    class_list: Sequence[int] = None,
    total_examples_per_class: int = None,
    preprocess_fn: Optional[Callable] = None,
    augmenter: Optional[Augmenter] = None,
    warmup: int = -1
)
```



<!-- Placeholder for "Used in" -->

Every <b>Sequence</b> must implement the <b>__getitem__</b> and the <b>__len__</b> methods.
If you want to modify your dataset between epochs you may implement
<b>on_epoch_end</b>.
The method <b>__getitem__</b> should return a complete batch.

#### Notes:



<b>Sequence</b> are a safer way to do multiprocessing. This structure guarantees
that the network will only train once
 on each sample per epoch which is not the case with generators.

##### # and <b>y_set</b> are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>dataset_name</b>
</td>
<td>
the name of the dataset to download and uses as
referenced in the TensorFlow catalog dataset page.
</td>
</tr><tr>
<td>
<b>x_key</b>
</td>
<td>
name of the dictonary key that contains the data to feed as
model input as referenced in the TensorFlow catalog dataset page.
Defaults to "image".
</td>
</tr><tr>
<td>
<b>y_key</b>
</td>
<td>
name of the dictonary key that contains the labels as
referenced in the TensorFlow catalog dataset page.
Defaults to "label".
</td>
</tr><tr>
<td>
<b>splits</b>
</td>
<td>
which dataset split(s) to use. Default
is ["train", "train"] Refersto the catalog page for
the list of available splits.
</td>
</tr><tr>
<td>
<b>examples_per_class_per_batch</b>
</td>
<td>
How many example of each class to
use per batch. Defaults to 2.
</td>
</tr><tr>
<td>
<b>steps_per_epoch</b>
</td>
<td>
How many steps/batches per epoch.
Defaults to 1000.
</td>
</tr><tr>
<td>
<b>class_list</b>
</td>
<td>
Filter the list of examples to only keep those who
belong to the supplied class list.
</td>
</tr><tr>
<td>
<b>total_examples_per_class</b>
</td>
<td>
Restrict the number of examples for EACH
class to total_examples_per_class if set. If not set, all the
available examples are selected. Defaults to None - no selection.
</td>
</tr><tr>
<td>
<b>preprocess_fn</b>
</td>
<td>
Preprocess function to apply to the dataset after
download e.g to resize images. Takes an x and a y.
Defaults to None.
</td>
</tr><tr>
<td>
<b>augmenter</b>
</td>
<td>
A function that takes a batch in and return a batch out.
Can alters the number of examples returned which in turn change the
batch_size used. Defaults to None.
</td>
</tr><tr>
<td>
<b>warmup</b>
</td>
<td>
Keep track of warmup epochs and let the augmenter knows
when the warmup is over by passing along with each batch data a
boolean <b>is_warmup</b>. See <b>self.get_examples()</b> Defaults to 0.
</td>
</tr>
</table>



## Methods

<h3 id="generate_batch"><code>generate_batch</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L122-L144">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>generate_batch(
    batch_id: int
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>, <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>]
</code></pre>

Generate a batch of data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
batch_id ([type]): [description]
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
x, y: batch
</td>
</tr>

</table>



<h3 id="get_examples"><code>get_examples</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/memory_samplers.py#L116-L132">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_examples(
    batch_id: int,
    num_classes: int,
    examples_per_class: int
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>, <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>]
</code></pre>

Get the set of examples that would be used to create a single batch.


#### Notes:

- before passing the batch data to TF, the sampler will call the
  augmenter function (if any) on the returned example.

- A batch_size = num_classes * example_per_class

- This function must be defined in the subclass.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch_id</b>
</td>
<td>
id of the batch in the epoch.
</td>
</tr><tr>
<td>
<b>num_classes</b>
</td>
<td>
How many class should be present in the examples.
</td>
</tr><tr>
<td>
<b>example_per_class</b>
</td>
<td>
How many example per class should be returned.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
x, y: batch of examples made of <b>num_classes</b> * <b>example_per_class</b>
</td>
</tr>

</table>



<h3 id="on_epoch_end"><code>on_epoch_end</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L107-L117">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_end() -> None
</code></pre>

Keep track of warmup epochs


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L119-L120">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    batch_id: int
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>, <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>]
</code></pre>

Gets batch at position <b>index</b>.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>index</b>
</td>
<td>
position of the batch in the Sequence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A batch
</td>
</tr>

</table>



<h3 id="__iter__"><code>__iter__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Create a generator that iterate over the Sequence.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L103-L105">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>

Return the number of batch per epoch




