# TFSimilarity.callbacks.Callback






Abstract base class used to build new callbacks.

```python
TFSimilarity.callbacks.Callback()
```



<!-- Placeholder for "Used in" -->

Callbacks can be passed to keras methods such as <b>fit</b>, <b>evaluate</b>, and
<b>predict</b> in order to hook into the various stages of the model training and
inference lifecycle.

To create a custom callback, subclass <b>keras.callbacks.Callback</b> and override
the method associated with the stage of interest. See
https://www.tensorflow.org/guide/keras/custom_callback for more information.

#### Example:



```
>>> training_finished = False
>>> class MyCallback(tf.keras.callbacks.Callback):
...   def on_train_end(self, logs=None):
...     global training_finished
...     training_finished = True
>>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
>>> model.compile(loss='mean_squared_error')
>>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
...           callbacks=[MyCallback()])
>>> assert training_finished == True
```

If you want to use <b>Callback</b> objects in a custom training loop:

1. You should pack all your callbacks into a single <b>callbacks.CallbackList</b>
   so they can all be called together.
2. You will need to manually call all the <b>on_*</b> methods at the appropriate
   locations in your loop. Like this:

   ```
   callbacks =  tf.keras.callbacks.CallbackList([...])
   callbacks.append(...)

   callbacks.on_train_begin(...)
   for epoch in range(EPOCHS):
     callbacks.on_epoch_begin(epoch)
     for i, data in dataset.enumerate():
       callbacks.on_train_batch_begin(i)
       batch_logs = model.train_step(data)
       callbacks.on_train_batch_end(i, batch_logs)
     epoch_logs = ...
     callbacks.on_epoch_end(epoch, epoch_logs)
   final_logs=...
   callbacks.on_train_end(final_logs)
   ```
The <b>logs</b> dictionary that callback methods
take as argument will contain keys for quantities relevant to
the current batch or epoch (see method-specific docstrings).



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>params</b>
</td>
<td>
Dict. Training parameters
(eg. verbosity, batch size, number of epochs...).
</td>
</tr><tr>
<td>
<b>model</b>
</td>
<td>
Instance of <b>keras.models.Model</b>.
Reference of the model being trained.
</td>
</tr>
</table>



## Methods

<h3 id="on_batch_begin">on_batch_begin</h3>

```python
on_batch_begin(
    batch, logs=None
)
```


A backwards compatibility alias for <b>on_train_batch_begin</b>.


<h3 id="on_batch_end">on_batch_end</h3>

```python
on_batch_end(
    batch, logs=None
)
```


A backwards compatibility alias for <b>on_train_batch_end</b>.


<h3 id="on_epoch_begin">on_epoch_begin</h3>

```python
on_epoch_begin(
    epoch, logs=None
)
```


Called at the start of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>epoch</b>
</td>
<td>
Integer, index of epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_epoch_end">on_epoch_end</h3>

```python
on_epoch_end(
    epoch, logs=None
)
```


Called at the end of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>epoch</b>
</td>
<td>
Integer, index of epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict, metric results for this training epoch, and for the
 validation epoch if validation is performed. Validation result keys
 are prefixed with <b>val_</b>. For training epoch, the values of the
<b>Model</b>'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
  0.7}`.
</td>
</tr>
</table>



<h3 id="on_predict_batch_begin">on_predict_batch_begin</h3>

```python
on_predict_batch_begin(
    batch, logs=None
)
```


Called at the beginning of a batch in <b>predict</b> methods.

Subclasses should override for any actions to run.

Note that if the <b>steps_per_execution</b> argument to <b>compile</b> in
<b>tf.keras.Model</b> is set to <b>N</b>, this method will only be called every <b>N</b>
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch</b>
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_predict_batch_end">on_predict_batch_end</h3>

```python
on_predict_batch_end(
    batch, logs=None
)
```


Called at the end of a batch in <b>predict</b> methods.

Subclasses should override for any actions to run.

Note that if the <b>steps_per_execution</b> argument to <b>compile</b> in
<b>tf.keras.Model</b> is set to <b>N</b>, this method will only be called every <b>N</b>
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch</b>
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Aggregated metric results up until this batch.
</td>
</tr>
</table>



<h3 id="on_predict_begin">on_predict_begin</h3>

```python
on_predict_begin(
    logs=None
)
```


Called at the beginning of prediction.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_predict_end">on_predict_end</h3>

```python
on_predict_end(
    logs=None
)
```


Called at the end of prediction.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_batch_begin">on_test_batch_begin</h3>

```python
on_test_batch_begin(
    batch, logs=None
)
```


Called at the beginning of a batch in <b>evaluate</b> methods.

Also called at the beginning of a validation batch in the <b>fit</b>
methods, if validation data is provided.

Subclasses should override for any actions to run.

Note that if the <b>steps_per_execution</b> argument to <b>compile</b> in
<b>tf.keras.Model</b> is set to <b>N</b>, this method will only be called every <b>N</b>
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch</b>
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_batch_end">on_test_batch_end</h3>

```python
on_test_batch_end(
    batch, logs=None
)
```


Called at the end of a batch in <b>evaluate</b> methods.

Also called at the end of a validation batch in the <b>fit</b>
methods, if validation data is provided.

Subclasses should override for any actions to run.

Note that if the <b>steps_per_execution</b> argument to <b>compile</b> in
<b>tf.keras.Model</b> is set to <b>N</b>, this method will only be called every <b>N</b>
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch</b>
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Aggregated metric results up until this batch.
</td>
</tr>
</table>



<h3 id="on_test_begin">on_test_begin</h3>

```python
on_test_begin(
    logs=None
)
```


Called at the beginning of evaluation or validation.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_test_end">on_test_end</h3>

```python
on_test_end(
    logs=None
)
```


Called at the end of evaluation or validation.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently the output of the last call to
<b>on_test_batch_end()</b> is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_batch_begin">on_train_batch_begin</h3>

```python
on_train_batch_begin(
    batch, logs=None
)
```


Called at the beginning of a training batch in <b>fit</b> methods.

Subclasses should override for any actions to run.

Note that if the <b>steps_per_execution</b> argument to <b>compile</b> in
<b>tf.keras.Model</b> is set to <b>N</b>, this method will only be called every <b>N</b>
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch</b>
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_batch_end">on_train_batch_end</h3>

```python
on_train_batch_end(
    batch, logs=None
)
```


Called at the end of a training batch in <b>fit</b> methods.

Subclasses should override for any actions to run.

Note that if the <b>steps_per_execution</b> argument to <b>compile</b> in
<b>tf.keras.Model</b> is set to <b>N</b>, this method will only be called every <b>N</b>
batches.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>batch</b>
</td>
<td>
Integer, index of batch within the current epoch.
</td>
</tr><tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Aggregated metric results up until this batch.
</td>
</tr>
</table>



<h3 id="on_train_begin">on_train_begin</h3>

```python
on_train_begin(
    logs=None
)
```


Called at the beginning of training.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently no data is passed to this argument for this method
but that may change in the future.
</td>
</tr>
</table>



<h3 id="on_train_end">on_train_end</h3>

```python
on_train_end(
    logs=None
)
```


Called at the end of training.

Subclasses should override for any actions to run.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>logs</b>
</td>
<td>
Dict. Currently the output of the last call to <b>on_epoch_end()</b>
is passed to this argument for this method but that may change in
the future.
</td>
</tr>
</table>



<h3 id="set_model">set_model</h3>

```python
set_model(
    model
)
```





<h3 id="set_params">set_params</h3>

```python
set_params(
    params
)
```







