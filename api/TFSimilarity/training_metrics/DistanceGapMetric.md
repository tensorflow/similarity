# TFSimilarity.training_metrics.DistanceGapMetric





Encapsulates metric logic and state.

Inherits From: [`Layer`](../../TFSimilarity/layers/Layer.md)

```python
TFSimilarity.training_metrics.DistanceGapMetric(
    distance, name=None, **kwargs
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>name</b>
</td>
<td>
(Optional) string name of the metric instance.
</td>
</tr><tr>
<td>
<b>dtype</b>
</td>
<td>
(Optional) data type of the metric result.
</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
Additional layer keywords arguments.
</td>
</tr>
</table>



#### Standalone usage:



```python
m = SomeMetric(...)
for input in ...:
  m.update_state(input)
print('Final result: ', m.result().numpy())
```

Usage with <b>compile()</b> API:

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.fit(dataset, epochs=10)
```

To be implemented by subclasses:
* <b>__init__()</b>: All state variables should be created in this method by
  calling <b>self.add_weight()</b> like: <b>self.var = self.add_weight(...)</b>
* <b>update_state()</b>: Has all updates to the state variables like:
  self.var.assign_add(...).
* <b>result()</b>: Computes and returns a value for the metric
  from the state variables.

Example subclass implementation:

```python
class BinaryTruePositives(tf.keras.metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_to(sample_weight, values.shape)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives
```

## Methods

<h3 id="reset_state">reset_state</h3>

```python
reset_state()
```


Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result">result</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/training_metrics/distance_metrics.py#L130-L131">View source</a>

```python
result()
```


Computes and returns the metric value tensor.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<h3 id="update_state">update_state</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/training_metrics/distance_metrics.py#L125-L128">View source</a>

```python
update_state(
    labels, embeddings, sample_weight
)
```


Accumulates statistics for the metric.

Note: This function is executed as a graph function in graph mode.
This means:
  a) Operations on the same resource are executed in textual order.
     This should make it easier to do things like add the updated
     value of a variable to another, for example.
  b) You don't need to worry about collecting the update ops to execute.
     All update ops added to the graph by this function will be executed.
  As a result, code should generally work the same way with graph or
  eager execution.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>*args</b>
</td>
<td>

</td>
</tr><tr>
<td>
<b>**kwargs</b>
</td>
<td>
A mini-batch of inputs to the Metric.
</td>
</tr>
</table>





