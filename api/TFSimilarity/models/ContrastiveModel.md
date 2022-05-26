# TFSimilarity.models.ContrastiveModel





`Model` groups layers into an object with training and inference features.

```python
TFSimilarity.models.ContrastiveModel(
    backbone: tf.keras.Model,
    projector: tf.keras.Model,
    predictor: Optional[tf.keras.Model] = None,
    algorithm: str = simsiam,
    **kwargs
) -> None
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>inputs</b>
</td>
<td>
The input(s) of the model: a <b>keras.Input</b> object or list of
<b>keras.Input</b> objects.
</td>
</tr><tr>
<td>
<b>outputs</b>
</td>
<td>
The output(s) of the model. See Functional API example below.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
String, the name of the model.
</td>
</tr>
</table>


There are two ways to instantiate a <b>Model</b>:

1 - With the "Functional API", where you start from <b>Input</b>,
you chain layer calls to specify the model's forward pass,
and finally you create your model from inputs and outputs:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

Note: Only dicts, lists, and tuples of input tensors are supported. Nested
inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the
intermediate tensors. This enables you to quickly extract sub-components
of the model.

#### Example:



```python
inputs = keras.Input(shape=(None, None, 3))
processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
pooling = keras.layers.GlobalAveragePooling2D()(conv)
feature = keras.layers.Dense(10)(pooling)

full_model = keras.Model(inputs, feature)
backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)
```

Note that the <b>backbone</b> and <b>activations</b> models are not
created with <b>keras.Input</b> objects, but with the tensors that are originated
from <b>keras.Inputs</b> objects. Under the hood, the layers and weights will
be shared across these models, so that user can train the <b>full_model</b>, and
use <b>backbone</b> or <b>activations</b> to do feature extraction.
The inputs and outputs of the model can be nested structures of tensors as
well, and the created models are standard Functional API models that support
all the existing APIs.

2 - By subclassing the <b>Model</b> class: in that case, you should define your
layers in <b>__init__()</b> and you should implement the model's forward pass
in <b>call()</b>.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
```

If you subclass <b>Model</b>, you can optionally have
a <b>training</b> argument (boolean) in <b>call()</b>, which you can use to specify
a different behavior in training and inference:

```python
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()
```

Once the model is created, you can config the model with losses and metrics
with <b>model.compile()</b>, train the model with <b>model.fit()</b>, or use the model
to do prediction with <b>model.predict()</b>.

## Methods

<h3 id="calibrate">calibrate</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L802-L872">View source</a>

```python
calibrate(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    thresholds_targets: MutableMapping[str, float] = {},
    k: int = 1,
    calibration_metric: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMetric.md">TFSimilarity.callbacks.ClassificationMetric```
</a>] = f1,
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>] = match_nearest,
    extra_metrics: MutableSequence[Union[str, ClassificationMetric]] = [precision, recall],
    rounding: int = 2,
    verbose: int = 1
) -> <a href="../../TFSimilarity/indexer/CalibrationResults.md">TFSimilarity.indexer.CalibrationResults```
</a>
```


Calibrate model thresholds using a test dataset.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
examples to use for the calibration.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
labels associated with the calibration examples.
</td>
</tr><tr>
<td>
<b>thresholds_targets</b>
</td>
<td>
Dict of performance targets to (if possible)
meet with respect to the <b>calibration_metric</b>.
</td>
</tr><tr>
<td>
<b>calibration_metric</b>
</td>
<td>
- [ClassificationMetric()](classification_metrics/overview.md) used
to evaluate the performance of the index.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
How many neighboors to use during the calibration.
Defaults to 1.
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
<i>'match_nearest', 'match_majority_vote'</i> or
ClassificationMatch object. Defines the classification matching,
e.g., match_nearest will count a True Positive if the query_label
is equal to the label of the nearest neighbor and the distance is
less than or equal to the distance threshold. Defaults to
'match_nearest'.
</td>
</tr><tr>
<td>
<b>extra_metrics</b>
</td>
<td>
List of additional
<b>tf.similarity.classification_metrics.ClassificationMetric()</b>
to compute and report. Defaults to ['precision', 'recall'].
</td>
</tr><tr>
<td>
<b>rounding</b>
</td>
<td>
Metric rounding. Default to 2 digits.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Be verbose and display calibration results.
Defaults to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
CalibrationResults containing the thresholds and cutpoints Dicts.
</td>
</tr>

</table>



<h3 id="create_index">create_index</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L600-L670">View source</a>

```python
create_index(
    distance: Union[<a href="../../TFSimilarity/distances/Distance.md">TFSimilarity.distances.Distance```
</a>, str] = cosine,
    search: Union[<a href="../../TFSimilarity/indexer/Search.md">TFSimilarity.indexer.Search```
</a>, str] = nmslib,
    kv_store: Union[<a href="../../TFSimilarity/indexer/Store.md">TFSimilarity.indexer.Store```
</a>, str] = memory,
    evaluator: Union[<a href="../../TFSimilarity/callbacks/Evaluator.md">TFSimilarity.callbacks.Evaluator```
</a>, str] = memory,
    embedding_output: Optional[int] = None,
    stat_buffer_size: int = 1000
) -> None
```


Create the model index to make embeddings searchable via KNN.

This method is normally called as part of <b>SimilarityModel.compile()</b>.
However, this method is provided if users want to define a custom index
outside of the <b>compile()</b> method.

NOTE: This method sets <b>SimilarityModel._index</b> and will replace any
existing index.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>distance</b>
</td>
<td>
Distance used to compute embeddings proximity. Defaults to
'auto'.
</td>
</tr><tr>
<td>
<b>kv_store</b>
</td>
<td>
How to store the indexed records.  Defaults to 'memory'.
</td>
</tr><tr>
<td>
<b>search</b>
</td>
<td>
Which <b>Search()</b> framework to use to perform KNN search.
Defaults to 'nmslib'.
</td>
</tr><tr>
<td>
<b>evaluator</b>
</td>
<td>
What type of <b>Evaluator()</b> to use to evaluate index
performance. Defaults to in-memory one.
</td>
</tr><tr>
<td>
<b>embedding_output</b>
</td>
<td>
Which model output head predicts the embeddings
that should be indexed. Defaults to None which is for single output
model. For multi-head model, the callee, usually the
<b>SimilarityModel()</b> class is responsible for passing the correct
one.
</td>
</tr><tr>
<td>
<b>stat_buffer_size</b>
</td>
<td>
Size of the sliding windows buffer used to compute
index performance. Defaults to 1000.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<b>ValueError</b>
</td>
<td>
Invalid search framework or key value store.
</td>
</tr>
</table>



<h3 id="evaluate_classification">evaluate_classification</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L996-L1103">View source</a>

```python
evaluate_classification(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    k: int = 1,
    extra_metrics: MutableSequence[Union[str, ClassificationMetric]] = [precision, recall],
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>] = match_nearest,
    verbose: int = 1
) -> DefaultDict[str, Dict[str, Union[str, np.ndarray]]]
```


Evaluate model classification matching on a given evaluation dataset.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Examples to be matched against the index.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
Label associated with the examples supplied.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
How many neighbors to use to perform the evaluation.
Defaults to 1.
</td>
</tr><tr>
<td>
<b>extra_metrics</b>
</td>
<td>
List of additional
<b>tf.similarity.classification_metrics.ClassificationMetric()</b> to
compute and report. Defaults to ['precision', 'recall'].
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
<i>'match_nearest', 'match_majority_vote'</i> or
ClassificationMatch object. Defines the classification matching,
e.g., match_nearest will count a True Positive if the query_label
is equal to the label of the nearest neighbor and the distance is
less than or equal to the distance threshold.

verbose (int, optional): Display results if set to 1 otherwise
results are returned silently. Defaults to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dictionary of (distance_metrics.md)[evaluation metrics]
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<b>IndexError</b>
</td>
<td>
Index must contain embeddings but is currently empty.
</td>
</tr><tr>
<td>
<b>ValueError</b>
</td>
<td>
Uncalibrated model: run model.calibration()")
</td>
</tr>
</table>



<h3 id="evaluate_retrieval">evaluate_retrieval</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L940-L994">View source</a>

```python
evaluate_retrieval(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    retrieval_metrics: Sequence[<a href="../../TFSimilarity/indexer/RetrievalMetric.md">TFSimilarity.indexer.RetrievalMetric```
</a>],
    verbose: int = 1
) -> Dict[str, np.ndarray]
```


Evaluate the quality of the index against a test dataset.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Examples to be matched against the index.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
Label associated with the examples supplied.
</td>
</tr><tr>
<td>
<b>retrieval_metrics</b>
</td>
<td>
List of
- [RetrievalMetric()](retrieval_metrics/overview.md) to compute.

verbose (int, optional): Display results if set to 1 otherwise
results are returned silently. Defaults to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dictionary of metric results where keys are the metric names and
values are the metrics values.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
<b>IndexError</b>
</td>
<td>
Index must contain embeddings but is currently empty.
</td>
</tr>
</table>



<h3 id="index">index</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L672-L714">View source</a>

```python
index(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a> = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None,
    build: bool = True,
    verbose: int = 1
)
```


Index data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Samples to index.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
class ids associated with the data if any. Defaults to None.
</td>
</tr><tr>
<td>
<b>store_data</b>
</td>
<td>
store the data associated with the samples in the key
value store. Defaults to True.
</td>
</tr><tr>
<td>
<b>build</b>
</td>
<td>
Rebuild the index after indexing. This is needed to make the
new samples searchable. Set it to false to save processing time
when calling indexing repeatidly without the need to search between
the indexing requests. Defaults to True.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Output indexing progress info. Defaults to 1.
</td>
</tr>
</table>



<h3 id="index_single">index_single</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L716-L759">View source</a>

```python
index_single(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a> = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None,
    build: bool = True,
    verbose: int = 1
)
```


Index data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Sample to index.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
class id associated with the data if any. Defaults to None.
</td>
</tr><tr>
<td>
<b>data</b>
</td>
<td>
store the data associated with the samples in the key
value store. Defaults to None.
</td>
</tr><tr>
<td>
<b>build</b>
</td>
<td>
Rebuild the index after indexing. This is needed to make the
new samples searchable. Set it to false to save processing time
when calling indexing repeatidly without the need to search between
the indexing requests. Defaults to True.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Output indexing progress info. Defaults to 1.
</td>
</tr>
</table>



<h3 id="index_size">index_size</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L1109-L1111">View source</a>

```python
index_size() -> int
```


Return the index size


<h3 id="index_summary">index_summary</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L798-L800">View source</a>

```python
index_summary()
```


Display index info summary.


<h3 id="load_index">load_index</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L1113-L1123">View source</a>

```python
load_index(
    filepath: str
)
```


Load Index data from a checkpoint and initialize underlying
structure with the reloaded data.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>path</b>
</td>
<td>
Directory where the checkpoint is located.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr>
</table>



<h3 id="lookup">lookup</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L761-L780">View source</a>

```python
lookup(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    k: int = 5,
    verbose: int = 1
) -> List[List[Lookup]]
```


Find the k closest matches in the index for a set of samples.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Samples to match.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
Number of nearest neighboors to lookup. Defaults to 5.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
display progress. Default to 1.
</td>
</tr>
</table>


Returns
    list of list of k nearest neighboors:
    List[List[Lookup]]

<h3 id="match">match</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L874-L938">View source</a>

```python
match(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    cutpoint=optimal,
    no_match_label=-1,
    k=1,
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>] = match_nearest,
    verbose=0
)
```


Match a set of examples against the calibrated index

For the match function to work, the index must be calibrated using
calibrate().

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Batch of examples to be matched against the index.
</td>
</tr><tr>
<td>
<b>cutpoint</b>
</td>
<td>
Which calibration threshold to use.
Defaults to 'optimal' which is the optimal F1 threshold computed
using calibrate().
</td>
</tr><tr>
<td>
<b>no_match_label</b>
</td>
<td>
Which label value to assign when there is no
match. Defaults to -1.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
How many neighboors to use during the calibration.
Defaults to 1.
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
<i>'match_nearest', 'match_majority_vote'</i> or
ClassificationMatch object. Defines the classification matching,
e.g., match_nearest will count a True Positive if the query_label
is equal to the label of the nearest neighbor and the distance is
less than or equal to the distance threshold.

verbose. Be verbose. Defaults to 0.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of class ids that matches for each supplied example
</td>
</tr>

</table>



#### Notes:

This function matches all the cutpoints at once internally as there
is little performance downside to do so and allows to do the
evaluation in a single go.


<h3 id="reset_index">reset_index</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L1105-L1107">View source</a>

```python
reset_index()
```


Reinitialize the index


<h3 id="save_index">save_index</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L1125-L1133">View source</a>

```python
save_index(
    filepath, compression=True
)
```


Save the index to disk


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>path</b>
</td>
<td>
directory where to save the index
</td>
</tr><tr>
<td>
<b>compression</b>
</td>
<td>
Store index data compressed. Defaults to True.
</td>
</tr>
</table>



<h3 id="single_lookup">single_lookup</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L782-L796">View source</a>

```python
single_lookup(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    k: int = 5
) -> List[<a href="../../TFSimilarity/indexer/Lookup.md">TFSimilarity.indexer.Lookup```
</a>]
```


Find the k closest matches in the index for a given sample.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>x</b>
</td>
<td>
Sample to match.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
Number of nearest neighboors to lookup. Defaults to 5.
</td>
</tr>
</table>


Returns
    list of the k nearest neigboors info:
    List[Lookup]

<h3 id="to_data_frame">to_data_frame</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/contrastive_model.py#L1135-L1145">View source</a>

```python
to_data_frame(
    num_items: int = 0
) -> <a href="../../TFSimilarity/indexer/PandasDataFrame.md">TFSimilarity.indexer.PandasDataFrame```
</a>
```


Export data as pandas dataframe


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
num_items (int, optional): Num items to export to the dataframe.
Defaults to 0 (unlimited).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>

<tr>
<td>
<b>pd.DataFrame</b>
</td>
<td>
a pandas dataframe.
</td>
</tr>
</table>





