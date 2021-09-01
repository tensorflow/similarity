# TFSimilarity.callbacks.SimilarityModel





Specialized Keras.Model which implement the core features needed for

Inherits From: [`Layer`](../../TFSimilarity/layers/Layer.md)


```python
TFSimilarity.callbacks.SimilarityModel(
    *args, **kwargs
)
```



<!-- Placeholder for "Used in" -->
metric learning. In particular, <b>SimilarityModel()</b> supports indexing,
searching and saving the embeddings predicted by the network.

All Similarity models classes derive from this class to benefits from those
core features.

## Methods

<h3 id="calibrate">calibrate</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L299-L361">View source</a>

```python
calibrate(
    x: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    thresholds_targets: MutableMapping[str, float] = {},
    k: int = 1,
    calibration_metric: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMetric.md">TFSimilarity.callbacks.ClassificationMetric```
</a>] = f1,
    matcher: Union[str, <a href="../../TFSimilarity/indexer/ClassificationMatch.md">TFSimilarity.indexer.ClassificationMatch```
</a>] = match_nearest,
    extra_metrics: MutableSequence[Union[str, ClassificationMetric]] = [precision, recall],
    rounding: int = 2,
    verbose: int = 1
)
```


Calibrate model thresholds using a test dataset.
FIXME: more detailed explaination.

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
Dict of performance targets to
(if possible) meet with respect to the <b>calibration_metric</b>.
</td>
</tr><tr>
<td>
<b>calibration_metric</b>
</td>
<td>
- [ClassificationMetric()](classification_metrics/overview.md)
used to evaluate the performance of the index.
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
ClassificationMatch object. Defines the classification
matching, e.g., match_nearest will count a True Positive if the
query_label is equal to the label of the nearest neighbor and
the distance is less than or equal to the distance threshold.
Defaults to 'match_nearest'.
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
Calibration results: <b><i>"cutpoints": {}, "thresholds": {}</i></b>
</td>
</tr>

</table>



<h3 id="evaluate_classification">evaluate_classification</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L412-L504">View source</a>

```python
evaluate_classification(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a>,
    k: int = 1,
    extra_metrics: MutableSequence[Union[str, ClassificationMetric]] = [precision, recall],
    matcher: Union[str, <a href="../../TFSimilarity/indexer/ClassificationMatch.md">TFSimilarity.indexer.ClassificationMatch```
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



<h3 id="index">index</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L218-L253">View source</a>

```python
index(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md">TFSimilarity.callbacks.IntTensor```
</a> = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None,
    build: bool = (True),
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



<h3 id="index_size">index_size</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L510-L512">View source</a>

```python
index_size() -> int
```


Return the index size


<h3 id="index_summary">index_summary</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L295-L297">View source</a>

```python
index_summary()
```


Display index info summary.


<h3 id="load_index">load_index</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L514-L524">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L255-L275">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L363-L410">View source</a>

```python
match(
    x: <a href="../../TFSimilarity/distances/FloatTensor.md">TFSimilarity.distances.FloatTensor```
</a>,
    cutpoint=optimal,
    no_match_label=-1,
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L506-L508">View source</a>

```python
reset_index()
```


Reinitialize the index


<h3 id="save_index">save_index</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L526-L534">View source</a>

```python
save_index(
    filepath, compression=(True)
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L277-L293">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L587-L597">View source</a>

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





