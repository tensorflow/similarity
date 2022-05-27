# TFSimilarity.indexer.Indexer





Indexing system that allows to efficiently find nearest embeddings

```python
TFSimilarity.indexer.Indexer(
    embedding_size: int,
    embedding_output: int = None,
    stat_buffer_size: int = 1000
) -> None
```



<!-- Placeholder for "Used in" -->
by indexing known embeddings and make them searchable using an
- [Approximate Nearest Neighbors Search]
(https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search implemented via the [<b>Search()</b>](search/overview.md) classes
and associated data lookup via the [<b>Store()</b>](stores/overview.md) classes.

The indexer allows to evaluate the quality of the constructed index and
calibrate the [SimilarityModel.match()](similarity_model.md) function via
the [<b>Evaluator()</b>](evaluators/overview.md) classes.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>embedding_size</b>
</td>
<td>
Size of the embeddings that will be stored.
It is usually equivalent to the size of the output layer.
</td>
</tr><tr>
<td>
<b>distance</b>
</td>
<td>
Distance used to compute embeddings proximity.
Defaults to 'cosine'.
</td>
</tr><tr>
<td>
<b>kv_store</b>
</td>
<td>
How to store the indexed records.
Defaults to 'memory'.
</td>
</tr><tr>
<td>
<b>search</b>
</td>
<td>
Which <b>Search()</b> framework to use to perform KNN
search. Defaults to 'nmslib'.
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
that should be indexed. Default to None which is for single output
model. For multi-head model, the callee, usually the
<b>SimilarityModel()</b> class is responsible for passing the correct
one.
</td>
</tr><tr>
<td>
<b>stat_buffer_size</b>
</td>
<td>
Size of the sliding windows
buffer used to compute index performance. Defaults to 1000.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
<b>ValueError</b>
</td>
<td>
Invalid search framework or key value store.
</td>
</tr>
</table>



## Methods

<h3 id="add">add</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L212-L245">View source</a>

```python
add(
    prediction: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    label: Optional[int] = None,
    data: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a> = None,
    build: bool = True,
    verbose: int = 1
)
```


Add a single embedding to the indexer


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>prediction</b>
</td>
<td>
TF similarity model prediction, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>label</b>
</td>
<td>
Label(s) associated with the
embedding. Defaults to None.
</td>
</tr><tr>
<td>
<b>data</b>
</td>
<td>
Input data associated with
the embedding. Defaults to None.
</td>
</tr><tr>
<td>
<b>build</b>
</td>
<td>
Rebuild the index after insertion.
Defaults to True. Set it to false if you would like to add
multiples batches/points and build it manually once after.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Display progress if set to 1.
Defaults to 1.
</td>
</tr>
</table>



<h3 id="batch_add">batch_add</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L247-L277">View source</a>

```python
batch_add(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    labels: Optional[Sequence[int]] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None,
    build: bool = True,
    verbose: int = 1
)
```


Add a batch of embeddings to the indexer


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>predictions</b>
</td>
<td>
TF similarity model predictions, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>labels</b>
</td>
<td>
label(s) associated with the embedding. Defaults to None.
</td>
</tr><tr>
<td>
<b>datas</b>
</td>
<td>
input data associated with the embedding. Defaults to None.
</td>
</tr><tr>
<td>
<b>build</b>
</td>
<td>
Rebuild the index after insertion.
Defaults to True. Set it to false if you would like to add
multiples batches/points and build it manually once after.
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
Display progress if set to 1. Defaults to 1.
</td>
</tr>
</table>



<h3 id="batch_lookup">batch_lookup</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L315-L377">View source</a>

```python
batch_lookup(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    k: int = 5,
    verbose: int = 1
) -> List[List[Lookup]]
```


Find the k closest matches for a set of embeddings


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>predictions</b>
</td>
<td>
TF similarity model predictions, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
Number of nearest neighbors to lookup. Defaults to 5.
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


Returns
    list of list of k nearest neighbors:
    List[List[Lookup]]

<h3 id="calibrate">calibrate</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L492-L583">View source</a>

```python
calibrate(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    target_labels: Sequence[int],
    thresholds_targets: MutableMapping[str, float],
    calibration_metric: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMetric.md">TFSimilarity.callbacks.ClassificationMetric```
</a>] = f1_score,
    k: int = 1,
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>] = match_nearest,
    extra_metrics: Sequence[Union[str, ClassificationMetric]] = [precision, recall],
    rounding: int = 2,
    verbose: int = 1
) -> <a href="../../TFSimilarity/indexer/CalibrationResults.md">TFSimilarity.indexer.CalibrationResults```
</a>
```


Calibrate model thresholds using a test dataset.

FIXME: more detailed explanation.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>predictions</b>
</td>
<td>
TF similarity model predictions, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>target_labels</b>
</td>
<td>
Sequence of the expected labels associated with the
embedded queries.
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
- [ClassificationMetric()](metrics/overview.md)
used to evaluate the performance of the index.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
How many neighbors to use during the calibration.
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
Be verbose and display calibration results. Defaults to 1.
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



<h3 id="evaluate_classification">evaluate_classification</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L425-L490">View source</a>

```python
evaluate_classification(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    target_labels: Sequence[int],
    distance_thresholds: Union[Sequence[float], <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>],
    metrics: Sequence[Union[str, ClassificationMetric]] = [f1],
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>] = match_nearest,
    k: int = 1,
    verbose: int = 1
) -> Dict[str, np.ndarray]
```


Evaluate the classification performance.

Compute the classification metrics given a set of queries, lookups, and
distance thresholds.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>predictions</b>
</td>
<td>
TF similarity model predictions, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>target_labels</b>
</td>
<td>
Sequence of expected labels for the lookups.
</td>
</tr><tr>
<td>
<b>distance_thresholds</b>
</td>
<td>
A 1D tensor denoting the distances points at
which we compute the metrics.
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
The set of classification metrics.
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
</td>
</tr><tr>
<td>
<b>distance_rounding</b>
</td>
<td>
How many digit to consider to
decide if the distance changed. Defaults to 8.
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Mapping from metric name to the list of values computed for each
distance threshold.
</td>
</tr>

</table>



<h3 id="evaluate_retrieval">evaluate_retrieval</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L380-L423">View source</a>

```python
evaluate_retrieval(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    target_labels: Sequence[int],
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
<b>predictions</b>
</td>
<td>
TF similarity model predictions, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>target_labels</b>
</td>
<td>
Sequence of the expected labels associated with the
embedded queries.
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



<h3 id="get_calibration_metric">get_calibration_metric</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L745-L746">View source</a>

```python
get_calibration_metric()
```





<h3 id="load">load</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L696-L743">View source</a>

``<b>python
@staticmethod</b>``

```python
load(
    path: Union[str, <a href="../../TFSimilarity/callbacks/Path.md">TFSimilarity.callbacks.Path```
</a>],
    verbose: int = 1
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Initialized index
</td>
</tr>

</table>



<h3 id="match">match</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L585-L660">View source</a>

```python
match(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    no_match_label: int = -1,
    k=1,
    matcher: Union[str, <a href="../../TFSimilarity/callbacks/ClassificationMatch.md">TFSimilarity.callbacks.ClassificationMatch```
</a>] = match_nearest,
    verbose: int = 1
) -> Dict[str, List[int]]
```


Match embeddings against the various cutpoints thresholds


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>predictions</b>
</td>
<td>
TF similarity model predictions, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>no_match_label</b>
</td>
<td>
What label value to assign when there is no match.
Defaults to -1.
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
</td>
</tr><tr>
<td>
<b>verbose</b>
</td>
<td>
display progression. Default to 1.
</td>
</tr>
</table>



#### Notes:


1. It is up to the [<b>SimilarityModel.match()</b>](similarity_model.md)
code to decide which of cutpoints results to use / show to the
users. This function returns all of them as there is little
performance downside to do so and it makes the code clearer
and simpler.

2. The calling function is responsible to return the list of class
matched to allows implementation to use additional criteria if they
choose to.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dict of cutpoint names mapped to lists of matches.
</td>
</tr>

</table>



<h3 id="print_stats">print_stats</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L775-L799">View source</a>

```python
print_stats()
```


display statistics in terminal friendly fashion


<h3 id="reset">reset</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L124-L126">View source</a>

```python
reset() -> None
```


Reinitialize the indexer


<h3 id="save">save</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L662-L694">View source</a>

```python
save(
    path: str, compression: bool = True
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

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L279-L313">View source</a>

```python
single_lookup(
    prediction: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    k: int = 5
) -> List[<a href="../../TFSimilarity/indexer/Lookup.md">TFSimilarity.indexer.Lookup```
</a>]
```


Find the k closest matches of a given embedding


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>prediction</b>
</td>
<td>
TF similarity model prediction, may be a multi-headed
output.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
Number of nearest neighbors to lookup. Defaults to 5.
</td>
</tr>
</table>


Returns
    list of the k nearest neighbors info:
    List[Lookup]

<h3 id="size">size</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L748-L750">View source</a>

```python
size() -> int
```


Return the index size


<h3 id="stats">stats</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L752-L773">View source</a>

```python
stats()
```


return index statistics


<h3 id="to_data_frame">to_data_frame</h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L801-L811">View source</a>

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







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
DATA<a id="DATA"></a>
</td>
<td>
<b>3</b>
</td>
</tr><tr>
<td>
DISTANCES<a id="DISTANCES"></a>
</td>
<td>
<b>1</b>
</td>
</tr><tr>
<td>
EMBEDDINGS<a id="EMBEDDINGS"></a>
</td>
<td>
<b>0</b>
</td>
</tr><tr>
<td>
LABELS<a id="LABELS"></a>
</td>
<td>
<b>2</b>
</td>
</tr><tr>
<td>
RANKS<a id="RANKS"></a>
</td>
<td>
<b>4</b>
</td>
</tr>
</table>

