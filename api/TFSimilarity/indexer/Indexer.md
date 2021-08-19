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
- [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search implemented via the [<b>Matcher()</b>](matchers/overview.md) classes
and associated data lookup via the [<b>Table()</b>](tables/overview.md) classes.

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
<b>table</b>
</td>
<td>
How to store the index records.
Defaults to 'memory'.
</td>
</tr><tr>
<td>
<b>matcher</b>
</td>
<td>
Which <b>Matcher()</b> framework to use to perfom KNN
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
Which model output head predicts
the embbedings that should be indexed. Default to None which is for
single output model. For multi-head model, the callee, usually the
<b>SimilarityModel()</b> class is responsible for passing the
correct one.
</td>
</tr><tr>
<td>
<b>stat_buffer_size</b>
</td>
<td>
Size of the sliding windows
buffer used to computer index performance. Defaults to 1000.
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
Invalid matcher or table.
</td>
</tr>
</table>



## Methods

<h3 id="add">add</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L189-L221">View source</a>

```python
add(
    prediction: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    label: Optional[int] = None,
    data: <a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a> = None,
    build: bool = (True),
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
TF similarity model prediction.
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
multiples batchs/points and build it manually once after.
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L223-L252">View source</a>

```python
batch_add(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    labels: Optional[List[Optional[int]]] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md">TFSimilarity.callbacks.Tensor```
</a>] = None,
    build: bool = (True),
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
TF similarity model predictions.
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
multiples batchs/points and build it manually once after.
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L288-L347">View source</a>

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
model predictions.
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
Be verbose. Defaults to 1.
</td>
</tr>
</table>


Returns
    list of list of k nearest neighboors:
    List[List[Lookup]]

<h3 id="calibrate">calibrate</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L378-L459">View source</a>

```python
calibrate(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    y: List[int],
    thresholds_targets: Dict[str, float],
    calibration_metric: Union[str, <a href="../../TFSimilarity/callbacks/EvalMetric.md">TFSimilarity.callbacks.EvalMetric```
</a>] = f1_score,
    k: int = 1,
    extra_metrics: List[Union[str, EvalMetric]] = [accuracy, recall],
    rounding: int = 2,
    verbose: int = 1
) -> Dict[str, Union[Dict[str, float], List[float]]]
```


Calibrate model thresholds using a test dataset.

FIXME: more detailed explaination.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
<b>predictions</b>
</td>
<td>
Test emebddings computed by the SimilarityModel.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
Expected labels for the nearest neighboors.
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
- [Metric()](metrics/overview.md) used to
evaluate the performance of the index.
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
<b>extra_metrics</b>
</td>
<td>
List of additional [Metric()](metrics/overview.md)
to compute and report.
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
Calibration results: <b><i>"cutpoints": {}, "thresholds": {}</i></b>
</td>
</tr>

</table>



<h3 id="evaluate">evaluate</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L350-L376">View source</a>

```python
evaluate(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    y: List[int],
    metrics: List[Union[str, EvalMetric]],
    k: int = 1,
    verbose: int = 1
) -> Dict[str, Union[float, int]]
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
Test emebddings computed by the SimilarityModel.
</td>
</tr><tr>
<td>
<b>y</b>
</td>
<td>
Expected labels for the nearest neighboors.
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
List of [Metric()](metrics/overview.md) to compute.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
How many neighboors to use during the evaluation. Defaults to 1.
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
Dictionary of metric results where keys are the metric names and
values are the metrics values.
</td>
</tr>

</table>



<h3 id="get_calibration_metric">get_calibration_metric</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L601-L602">View source</a>

```python
get_calibration_metric()
```





<h3 id="load">load</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L555-L599">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L461-L519">View source</a>

```python
match(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md">TFSimilarity.callbacks.FloatTensor```
</a>,
    no_match_label: int = -1,
    verbose: int = 1
) -> Dict[str, List[int]]
```


Match embeddings against the various cutpoints thresholds


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr class="alt">
<td colspan="2">
predictions (FloatTensor): embeddings

no_match_label (int, optional): What label value to assign when
there is no match. Defaults to -1.

verbose (int): display progression. Default to 1.
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
matched to allows implementation to use additional criterias
if they choose to.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dict of matches list keyed by cutpoint names.
</td>
</tr>

</table>



<h3 id="print_stats">print_stats</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L631-L655">View source</a>

```python
print_stats()
```


display statistics in terminal friendly fashion


<h3 id="reset">reset</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L103-L105">View source</a>

```python
reset() -> None
```


Reinitialize the indexer


<h3 id="save">save</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L521-L553">View source</a>

```python
save(
    path: str,
    compression: bool = (True)
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

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L254-L286">View source</a>

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
model prediction.
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

<h3 id="size">size</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L604-L606">View source</a>

```python
size() -> int
```


Return the index size


<h3 id="stats">stats</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L608-L629">View source</a>

```python
stats()
```


return index statistics


<h3 id="to_data_frame">to_data_frame</h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L657-L667">View source</a>

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

