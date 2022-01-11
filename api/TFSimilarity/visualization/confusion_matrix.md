# TFSimilarity.visualization.confusion_matrix





Plot confusion matrix

```python
TFSimilarity.visualization.confusion_matrix(
    normalize: bool = True,
    title: str = Confusion matrix,
    cmap: str = Blues,
    show: bool = True
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
<b>y_pred</b>
</td>
<td>
Model prediction returned by <b>model.match()</b>
</td>
</tr><tr>
<td>
<b>y_true</b>
</td>
<td>
Expected class_id.
</td>
</tr><tr>
<td>
<b>normalize</b>
</td>
<td>
Normalizes matrix values between 0 and 1.
Defaults to True.
</td>
</tr><tr>
<td>
<b>labels</b>
</td>
<td>
List of class string label to display instead of the class
numerical ids. Defaults to None.
</td>
</tr><tr>
<td>
<b>title</b>
</td>
<td>
Title of the confusion matrix. Defaults to 'Confusion matrix'.
</td>
</tr><tr>
<td>
<b>cmap</b>
</td>
<td>
Color schema as CMAP. Defaults to 'Blues'.
</td>
</tr><tr>
<td>
<b>show</b>
</td>
<td>
If the plot is going to be shown or not. Defaults to True.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tuple containing the plot and confusion matrix.
</td>
</tr>

</table>

