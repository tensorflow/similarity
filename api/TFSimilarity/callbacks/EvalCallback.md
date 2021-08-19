
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.callbacks.EvalCallback" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>
# TFSimilarity.callbacks.EvalCallback
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/callbacks.py#L14-L97">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Epoch end evaluation callback that build a test index and evaluate
Inherits From: [`Callback`](../../TFSimilarity/callbacks/Callback.md)
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.callbacks.EvalCallback(
    queries: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    query_labels: List[int],
    targets: <a href="../../TFSimilarity/callbacks/Tensor.md"><code>TFSimilarity.callbacks.Tensor</code></a>,
    target_labels: List[int],
    distance: str = &#x27;cosine&#x27;,
    metrics: List[Union[str, EvalMetric]] = [&#x27;accuracy&#x27;, &#x27;mean_rank&#x27;],
    tb_logdir: str = None,
    k: int = 1
)
</code></pre>

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
`queries`
</td>
<td>
Test examples that will be tested against the built index.
</td>
</tr><tr>
<td>
`query_labels`
</td>
<td>
Queries nearest neighboors expected labels.
</td>
</tr><tr>
<td>
`targets`
</td>
<td>
Examples that are indexed.
</td>
</tr><tr>
<td>
`target_labels`
</td>
<td>
Target examples labels.
</td>
</tr><tr>
<td>
`distance`
</td>
<td>
Distance function used to compute pairwise distance
between examples embeddings.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
List of [EvalMetrics](eval_metrics.md) to be computed
during the evaluation. Defaults to ['accuracy', 'mean_rank'].
embedding to evaluate.
</td>
</tr><tr>
<td>
`tb_logdir`
</td>
<td>
Where to write TensorBoard logs. Defaults to None.
</td>
</tr><tr>
<td>
`k`
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



