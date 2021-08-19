
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.indexer.Indexer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="batch_add"/>
<meta itemprop="property" content="batch_lookup"/>
<meta itemprop="property" content="calibrate"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="get_calibration_metric"/>
<meta itemprop="property" content="load"/>
<meta itemprop="property" content="match"/>
<meta itemprop="property" content="print_stats"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="single_lookup"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="stats"/>
<meta itemprop="property" content="to_data_frame"/>
<meta itemprop="property" content="DATA"/>
<meta itemprop="property" content="DISTANCES"/>
<meta itemprop="property" content="EMBEDDINGS"/>
<meta itemprop="property" content="LABELS"/>
<meta itemprop="property" content="RANKS"/>
</div>
# TFSimilarity.indexer.Indexer
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L24-L671">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Indexing system that allows to efficiently find nearest embeddings
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.indexer.Indexer(
    embedding_size: int,
    distance: Union[<a href="../../TFSimilarity/distances/Distance.md"><code>TFSimilarity.distances.Distance</code></a>, str] = &#x27;cosine&#x27;,
    matcher: Union[<a href="../../TFSimilarity/indexer/Matcher.md"><code>TFSimilarity.indexer.Matcher</code></a>, str] = &#x27;nmslib&#x27;,
    table: Union[<a href="../../TFSimilarity/indexer/Table.md"><code>TFSimilarity.indexer.Table</code></a>, str] = &#x27;memory&#x27;,
    evaluator: Union[<a href="../../TFSimilarity/evaluators/Evaluator.md"><code>TFSimilarity.evaluators.Evaluator</code></a>, str] = &#x27;memory&#x27;,
    embedding_output: int = None,
    stat_buffer_size: int = 1000
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->
by indexing known embeddings and make them searchable using an
[Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search implemented via the [`Matcher()`](matchers/overview.md) classes
and associated data lookup via the [`Table()`](tables/overview.md) classes.
The indexer allows to evaluate the quality of the constructed index and
calibrate the [SimilarityModel.match()](similarity_model.md) function via
the [`Evaluator()`](evaluators/overview.md) classes.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`embedding_size`
</td>
<td>
Size of the embeddings that will be stored.
It is usually equivalent to the size of the output layer.
</td>
</tr><tr>
<td>
`distance`
</td>
<td>
Distance used to compute embeddings proximity.
Defaults to 'cosine'.
</td>
</tr><tr>
<td>
`table`
</td>
<td>
How to store the index records.
Defaults to 'memory'.
</td>
</tr><tr>
<td>
`matcher`
</td>
<td>
Which `Matcher()` framework to use to perfom KNN
search. Defaults to 'nmslib'.
</td>
</tr><tr>
<td>
`evaluator`
</td>
<td>
What type of `Evaluator()` to use to evaluate index
performance. Defaults to in-memory one.
</td>
</tr><tr>
<td>
`embedding_output`
</td>
<td>
Which model output head predicts
the embbedings that should be indexed. Default to None which is for
single output model. For multi-head model, the callee, usually the
`SimilarityModel()` class is responsible for passing the
correct one.
</td>
</tr><tr>
<td>
`stat_buffer_size`
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
`ValueError`
</td>
<td>
Invalid matcher or table.
</td>
</tr>
</table>

## Methods
<h3 id="add"><code>add</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L189-L221">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add(
    prediction: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    label: Optional[int] = None,
    data: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a> = None,
    build: bool = (True),
    verbose: int = 1
)
</code></pre>
Add a single embedding to the indexer

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`prediction`
</td>
<td>
TF similarity model prediction.
</td>
</tr><tr>
<td>
`label`
</td>
<td>
Label(s) associated with the
embedding. Defaults to None.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
Input data associated with
the embedding. Defaults to None.
</td>
</tr><tr>
<td>
`build`
</td>
<td>
Rebuild the index after insertion.
Defaults to True. Set it to false if you would like to add
multiples batchs/points and build it manually once after.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Display progress if set to 1.
Defaults to 1.
</td>
</tr>
</table>

<h3 id="batch_add"><code>batch_add</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L223-L252">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_add(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    labels: Optional[List[Optional[int]]] = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>] = None,
    build: bool = (True),
    verbose: int = 1
)
</code></pre>
Add a batch of embeddings to the indexer

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`predictions`
</td>
<td>
TF similarity model predictions.
</td>
</tr><tr>
<td>
`labels`
</td>
<td>
label(s) associated with the embedding. Defaults to None.
</td>
</tr><tr>
<td>
`datas`
</td>
<td>
input data associated with the embedding. Defaults to None.
</td>
</tr><tr>
<td>
`build`
</td>
<td>
Rebuild the index after insertion.
Defaults to True. Set it to false if you would like to add
multiples batchs/points and build it manually once after.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Display progress if set to 1. Defaults to 1.
</td>
</tr>
</table>

<h3 id="batch_lookup"><code>batch_lookup</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L288-L347">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>batch_lookup(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    k: int = 5,
    verbose: int = 1
) -> List[List[Lookup]]
</code></pre>
Find the k closest matches for a set of embeddings

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`predictions`
</td>
<td>
model predictions.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
Number of nearest neighboors to lookup. Defaults to 5.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Be verbose. Defaults to 1.
</td>
</tr>
</table>

Returns
    list of list of k nearest neighboors:
    List[List[Lookup]]
<h3 id="calibrate"><code>calibrate</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L378-L459">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>calibrate(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    y: List[int],
    thresholds_targets: Dict[str, float],
    calibration_metric: Union[str, <a href="../../TFSimilarity/callbacks/EvalMetric.md"><code>TFSimilarity.callbacks.EvalMetric</code></a>] = &#x27;f1_score&#x27;,
    k: int = 1,
    extra_metrics: List[Union[str, EvalMetric]] = [&#x27;accuracy&#x27;, &#x27;recall&#x27;],
    rounding: int = 2,
    verbose: int = 1
) -> Dict[str, Union[Dict[str, float], List[float]]]
</code></pre>
Calibrate model thresholds using a test dataset.
FIXME: more detailed explaination.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`predictions`
</td>
<td>
Test emebddings computed by the SimilarityModel.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Expected labels for the nearest neighboors.
</td>
</tr><tr>
<td>
`thresholds_targets`
</td>
<td>
Dict of performance targets to (if possible)
meet with respect to the `calibration_metric`.
</td>
</tr><tr>
<td>
`calibration_metric`
</td>
<td>
[Metric()](metrics/overview.md) used to
evaluate the performance of the index.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
How many neighboors to use during the calibration.
Defaults to 1.
</td>
</tr><tr>
<td>
`extra_metrics`
</td>
<td>
List of additional [Metric()](metrics/overview.md)
to compute and report.
</td>
</tr><tr>
<td>
`rounding`
</td>
<td>
Metric rounding. Default to 2 digits.
</td>
</tr><tr>
<td>
`verbose`
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
Calibration results: `{"cutpoints": {}, "thresholds": {}}`
</td>
</tr>
</table>

<h3 id="evaluate"><code>evaluate</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L350-L376">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    y: List[int],
    metrics: List[Union[str, EvalMetric]],
    k: int = 1,
    verbose: int = 1
) -> Dict[str, Union[float, int]]
</code></pre>
Evaluate the quality of the index against a test dataset.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`predictions`
</td>
<td>
Test emebddings computed by the SimilarityModel.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Expected labels for the nearest neighboors.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
List of [Metric()](metrics/overview.md) to compute.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
How many neighboors to use during the evaluation. Defaults to 1.
</td>
</tr><tr>
<td>
`verbose`
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

<h3 id="get_calibration_metric"><code>get_calibration_metric</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L601-L602">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_calibration_metric()
</code></pre>


<h3 id="load"><code>load</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L555-L599">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>load(
    path: Union[str, <a href="../../TFSimilarity/callbacks/Path.md"><code>TFSimilarity.callbacks.Path</code></a>],
    verbose: int = 1
)
</code></pre>
Load Index data from a checkpoint and initialize underlying
structure with the reloaded data.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`path`
</td>
<td>
Directory where the checkpoint is located.
</td>
</tr><tr>
<td>
`verbose`
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

<h3 id="match"><code>match</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L461-L519">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>match(
    predictions: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    no_match_label: int = -1,
    verbose: int = 1
) -> Dict[str, List[int]]
</code></pre>
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

1. It is up to the [`SimilarityModel.match()`](similarity_model.md)
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

<h3 id="print_stats"><code>print_stats</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L631-L655">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>print_stats()
</code></pre>
display statistics in terminal friendly fashion

<h3 id="reset"><code>reset</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L103-L105">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset() -> None
</code></pre>
Reinitialize the indexer

<h3 id="save"><code>save</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L521-L553">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    path: str,
    compression: bool = (True)
)
</code></pre>
Save the index to disk

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`path`
</td>
<td>
directory where to save the index
</td>
</tr><tr>
<td>
`compression`
</td>
<td>
Store index data compressed. Defaults to True.
</td>
</tr>
</table>

<h3 id="single_lookup"><code>single_lookup</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L254-L286">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>single_lookup(
    prediction: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    k: int = 5
) -> List[<a href="../../TFSimilarity/indexer/Lookup.md"><code>TFSimilarity.indexer.Lookup</code></a>]
</code></pre>
Find the k closest matches of a given embedding

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`prediction`
</td>
<td>
model prediction.
</td>
</tr><tr>
<td>
`k`
</td>
<td>
Number of nearest neighboors to lookup. Defaults to 5.
</td>
</tr>
</table>

Returns
    list of the k nearest neigboors info:
    List[Lookup]
<h3 id="size"><code>size</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L604-L606">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>size() -> int
</code></pre>
Return the index size

<h3 id="stats"><code>stats</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L608-L629">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stats()
</code></pre>
return index statistics

<h3 id="to_data_frame"><code>to_data_frame</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/indexer.py#L657-L667">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_data_frame(
    num_items: int = 0
) -> <a href="../../TFSimilarity/indexer/PandasDataFrame.md"><code>TFSimilarity.indexer.PandasDataFrame</code></a>
</code></pre>
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
`pd.DataFrame`
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
`3`
</td>
</tr><tr>
<td>
DISTANCES<a id="DISTANCES"></a>
</td>
<td>
`1`
</td>
</tr><tr>
<td>
EMBEDDINGS<a id="EMBEDDINGS"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
LABELS<a id="LABELS"></a>
</td>
<td>
`2`
</td>
</tr><tr>
<td>
RANKS<a id="RANKS"></a>
</td>
<td>
`4`
</td>
</tr>
</table>
