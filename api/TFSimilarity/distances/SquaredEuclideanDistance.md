
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.distances.SquaredEuclideanDistance" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="get_config"/>
</div>
# TFSimilarity.distances.SquaredEuclideanDistance
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L143-L170">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Compute pairwise squared Euclidean distance.
Inherits From: [`Distance`](../../TFSimilarity/distances/Distance.md), [`ABC`](../../TFSimilarity/distances/ABC.md)
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.distances.SquaredEuclideanDistance()
</code></pre>

<!-- Placeholder for "Used in" -->
The [Sequared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance) is
an distance that varies from 0 (similar) to infinity (dissimilar).
## Methods
<h3 id="call"><code>call</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L152-L170">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf.function</code>
<code>call(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>
) -> <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>
</code></pre>
Compute pairwise distances for a given batch of embeddings.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>
<tr>
<td>
`embeddings`
</td>
<td>
Embeddings to compute the pairwise one.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr>
<td>
`FloatTensor`
</td>
<td>
Pairwise distance tensor.
</td>
</tr>
</table>

<h3 id="get_config"><code>get_config</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L34-L35">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>


<h3 id="__call__"><code>__call__</code></h3>
<a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/distances.py#L28-L29">View source</a>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    embeddings: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>
)
</code></pre>
Call self as a function.


