# TFSimilarity.models.SimilarityModel





Specialized Keras.Model which implement the core features needed for

```python
TFSimilarity.models.SimilarityModel(
    *args, **kwargs
)
```



<!-- Placeholder for "Used in" -->
metric learning. In particular, <b>SimilarityModel()</b> supports indexing,
searching and saving the embeddings predicted by the network.

All Similarity models classes derive from this class to benefits from those
core features.

## Methods

<h3 id="calibrate"><code>calibrate</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L275-L325">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>calibrate(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    thresholds_targets: Dict[str, float] = {},
    k: int = 1,
    calibration_metric: Union[str, <a href="../../TFSimilarity/callbacks/EvalMetric.md"><code>TFSimilarity.callbacks.EvalMetric</code></a>] = &#x27;f1_score&#x27;,
    extra_metrics: List[Union[str, EvalMetric]] = [&#x27;accuracy&#x27;, &#x27;recall&#x27;],
    rounding: int = 2,
    verbose: int = 1
)
</code></pre>

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
Expected labels for the nearest neighboors.
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
List of additional [Metric()](
metrics/overview.md) to compute and report.
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



<h3 id="evaluate_matching"><code>evaluate_matching</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L376-L448">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate_matching(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    k: int = 1,
    extra_metrics: List[Union[EvalMetric, str]] = [&#x27;accuracy&#x27;, &#x27;recall&#x27;],
    verbose: int = 1
)
</code></pre>

Evaluate model matching accuracy on a given evaluation dataset.


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
How many neigboors to use to perform the evaluation.
Defaults to 1.
</td>
</tr><tr>
<td>
<b>extra_metrics</b>
</td>
<td>
Additional (distance_metrics.mde)[distance metrics]
to be computed during the evaluation. Defaut to accuracy and
recall.

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
Dictionnary of (distance_metrics.md)[evaluation metrics]
</td>
</tr>

</table>



<h3 id="index"><code>index</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L196-L229">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>index(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a> = None,
    data: Optional[<a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>] = None,
    build: bool = (True),
    verbose: int = 1
)
</code></pre>

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
store the data associated with the samples in Table.
Defaults to True.
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



<h3 id="index_size"><code>index_size</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L454-L456">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>index_size() -> int
</code></pre>

Return the index size


<h3 id="index_summary"><code>index_summary</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L271-L273">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>index_summary()
</code></pre>

Display index info summary.


<h3 id="load_index"><code>load_index</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L458-L468">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_index(
    filepath: str
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



<h3 id="lookup"><code>lookup</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L231-L251">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lookup(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    k: int = 5,
    verbose: int = 1
) -> List[List[Lookup]]
</code></pre>

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

<h3 id="match"><code>match</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L327-L374">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>match(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    cutpoint=&#x27;optimal&#x27;,
    no_match_label=-1,
    verbose=0
)
</code></pre>

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


<h3 id="reset_index"><code>reset_index</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L450-L452">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_index()
</code></pre>

Reinitialize the index


<h3 id="save_index"><code>save_index</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L470-L478">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_index(
    filepath, compression=(True)
)
</code></pre>

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



<h3 id="single_lookup"><code>single_lookup</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L253-L269">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>single_lookup(
    x: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    k: int = 5
) -> List[<a href="../../TFSimilarity/indexer/Lookup.md"><code>TFSimilarity.indexer.Lookup</code></a>]
</code></pre>

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

<h3 id="to_data_frame"><code>to_data_frame</code></h3>

<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/models/similarity_model.py#L531-L541">View source</a>

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
<b>pd.DataFrame</b>
</td>
<td>
a pandas dataframe.
</td>
</tr>
</table>





