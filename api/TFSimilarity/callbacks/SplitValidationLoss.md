# TFSimilarity.callbacks.SplitValidationLoss





A split validation callback.

Inherits From: [`Callback`](../../TFSimilarity/callbacks/Callback.md)

```python
TFSimilarity.callbacks.SplitValidationLoss(
    query_labels: Sequence[int],
    target_labels: Sequence[int],
    distance: str = cosine,
    metrics: Sequence[Union[str, ClassificationMetric]] = [binary_accuracy, f1score],
    tb_logdir: str = None,
    k=1
)
```



<!-- Placeholder for "Used in" -->

This callback will split the validation data into two sets.

    1) The set of classes seen during training.
    2) The set of classes not seen during training.

The callback will then compute a separate validation for each split.

This is useful for separately tracking the validation loss on the seen and
unseen classes and may provide insight into how well the embedding will
generalize to new classes.

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
Queries nearest neighbors expected labels.
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
<b>known_classes</b>
</td>
<td>
The set of classes seen during training.
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
List of
'tf.similarity.classification_metrics.ClassificationMetric()` to
compute during the evaluation. Defaults to ['binary_accuracy',
'f1score'].
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
How many neighbors to retrieve for evaluation. Defaults to 1.
</td>
</tr>
</table>



## Methods

<h3 id="set_model">set_model</h3>

```python
set_model(
    model
)
```





<h3 id="set_params">set_params</h3>

```python
set_params(
    params
)
```







