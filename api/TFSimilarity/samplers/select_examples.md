# TFSimilarity.samplers.select_examples
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/samplers/utils.py#L10-L70">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Randomly select at most N examples per class
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.samplers.select_examples(
    x: <a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>,
    y: <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>,
    class_list: Sequence[int] = None,
    num_examples_per_class: int = None
) -> Tuple[<a href="../../TFSimilarity/callbacks/FloatTensor.md"><code>TFSimilarity.callbacks.FloatTensor</code></a>, <a href="../../TFSimilarity/callbacks/IntTensor.md"><code>TFSimilarity.callbacks.IntTensor</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr>
<td>
`x`
</td>
<td>
A 2-D Tensor containing the data.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
A 1-D Tensor containing the labels.
</td>
</tr><tr>
<td>
`class_list`
</td>
<td>
Filter the list of examples to only keep thoses those who
belong to the supplied class list. In no class is supplied, keep
examples for all the classes. Default to None - keep all the examples.
</td>
</tr><tr>
<td>
`num_examples_per_class`
</td>
<td>
Restrict the number of examples for EACH
class to num_examples_per_class if set. If not set, all the available
examples are selected. Defaults to None - no selection.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tuple containing the subset of x and y.
</td>
</tr>
</table>
