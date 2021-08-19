# TFSimilarity.callbacks.EvalCallback





Epoch end evaluation callback that build a test index and evaluate

Inherits From: [`Callback`](../../TFSimilarity/callbacks/Callback.md)

```python
TFSimilarity.callbacks.EvalCallback(
```

    query_labels: List[int],
```

    target_labels: List[int],
    distance: str = cosine,
    metrics: List[Union[str, EvalMetric]] = [accuracy, mean_rank],
    tb_logdir: str = None,
    k: int = 1
)
```



<!-- Placeholder for "Used in" -->
model performance on it.

This evaluation only run at epoch_end as it is computationally very
expensive.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>queries</b>
</td>
<td>
Test examples that will be tested against the built index.
</td>
</tr><tr>
<td>
<b>query_labels</b>
</td>
<td>
Queries nearest neighboors expected labels.
</td>
</tr><tr>
<td>
<b>targets</b>
</td>
<td>
Examples that are indexed.
</td>
</tr><tr>
<td>
<b>target_labels</b>
</td>
<td>
Target examples labels.
</td>
</tr><tr>
<td>
<b>distance</b>
</td>
<td>
Distance function used to compute pairwise distance
between examples embeddings.
</td>
</tr><tr>
<td>
<b>metrics</b>
</td>
<td>
List of [EvalMetrics](eval_metrics.md) to be computed
during the evaluation. Defaults to ['accuracy', 'mean_rank'].
embedding to evaluate.
</td>
</tr><tr>
<td>
<b>tb_logdir</b>
</td>
<td>
Where to write TensorBoard logs. Defaults to None.
</td>
</tr><tr>
<td>
<b>k</b>
</td>
<td>
How many neigboors to retrive for evaluation. Defaults to 1.
</td>
</tr>
</table>



## Methods

<h3 id="set_model"><code>set_model</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>




<h3 id="set_params"><code>set_params</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






