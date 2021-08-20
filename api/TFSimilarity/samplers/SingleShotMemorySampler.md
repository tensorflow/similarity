# TFSimilarity.samplers.SingleShotMemorySampler





Base object for fitting to a sequence of data, such as a dataset.

Inherits From: [`Sampler`](../../TFSimilarity/metrics/Sampler.md)

```python
TFSimilarity.samplers.SingleShotMemorySampler(
    augmenter: Augmenter,
    class_per_batch: int,
    steps_per_epoch: int = 1000,
    warmup: int = -1
) -> None
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
<b>x</b>
</td>
<td>
Input data. The sampler assumes that each element of X is from a
distinct class.
</td>
</tr><tr>
<td>
<b>augmenter</b>
</td>
<td>
A function that takes a batch of single examples and
return a batch out with additional examples per class.
</td>
</tr><tr>
<td>
<b>steps_per_epoch</b>
</td>
<td>
How many steps/batch per epoch. Defaults to 1000.
</td>
</tr><tr>
<td>
<b>class_per_batch</b>
</td>
<td>
effectively the number of element to pass to the
augmenter for each batch request in the single shot setting.
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

<h3 id="generate_batch">generate_batch</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L122-L144">View source</a>

```python
generate_batch(
    batch_id: int
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor``<b>
</a>, <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor</b>``
</a>]
```


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



<h3 id="get_examples">get_examples</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/memory_samplers.py#L187-L202">View source</a>

```python
get_examples(
    batch_id: int,
    num_classes: int,
    example_per_class: int
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor``<b>
</a>, <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor</b>``
</a>]
```


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



<h3 id="on_epoch_end">on_epoch_end</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L107-L117">View source</a>

```python
on_epoch_end() -> None
```


Keep track of warmup epochs


<h3 id="__getitem__">__getitem__</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L119-L120">View source</a>

```python
__getitem__(
    batch_id: int
) -> Tuple[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor``<b>
</a>, <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor</b>``
</a>]
```


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



<h3 id="__iter__">__iter__</h3>

```python
__iter__()
```


Create a generator that iterate over the Sequence.


<h3 id="__len__">__len__</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/samplers.py#L103-L105">View source</a>

```python
__len__() -> int
```


Return the number of batch per epoch




