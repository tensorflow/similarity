
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.callbacks.FloatTensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="count"/>
<meta itemprop="property" content="extend"/>
<meta itemprop="property" content="index"/>
<meta itemprop="property" content="insert"/>
<meta itemprop="property" content="pop"/>
<meta itemprop="property" content="remove"/>
<meta itemprop="property" content="reverse"/>
<meta itemprop="property" content="sort"/>
</div>
# TFSimilarity.callbacks.FloatTensor
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity/types.py#L59-L60">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>

Float tensor 
Inherits From: [`Tensor`](../../TFSimilarity/callbacks/Tensor.md)
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.distances.FloatTensor`, `TFSimilarity.indexer.FloatTensor`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.callbacks.FloatTensor(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`__iter__`
</td>
<td>
Implement iter(self).
</td>
</tr><tr>
<td>
`__len__`
</td>
<td>
Return len(self).
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
</td>
</tr>
</table>


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`__iter__`
</td>
<td>
Implement iter(self).
</td>
</tr><tr>
<td>
`__len__`
</td>
<td>
Return len(self).
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
</td>
</tr>
</table>

## Methods
<h3 id="append"><code>append</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>append(
    object, /
)
</code></pre>
Append object to the end of the list.

<h3 id="clear"><code>clear</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>
Remove all items from list.

<h3 id="copy"><code>copy</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy()
</code></pre>
Return a shallow copy of the list.

<h3 id="count"><code>count</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>count(
    value, /
)
</code></pre>
Return number of occurrences of value.

<h3 id="extend"><code>extend</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>extend(
    iterable, /
)
</code></pre>
Extend list by appending elements from the iterable.

<h3 id="index"><code>index</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>index(
    value, start, stop, /
)
</code></pre>
Return first index of value.
Raises ValueError if the value is not present.
<h3 id="insert"><code>insert</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert(
    index, object, /
)
</code></pre>
Insert object before index.

<h3 id="pop"><code>pop</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pop(
    index, /
)
</code></pre>
Remove and return item at index (default last).
Raises IndexError if list is empty or index is out of range.
<h3 id="remove"><code>remove</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove(
    value, /
)
</code></pre>
Remove first occurrence of value.
Raises ValueError if the value is not present.
<h3 id="reverse"><code>reverse</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reverse()
</code></pre>
Reverse *IN PLACE*.

<h3 id="sort"><code>sort</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sort(
    *, key=None, reverse=False
)
</code></pre>
Sort the list in ascending order and return None.
The sort is in-place (i.e. the list itself is modified) and stable (i.e. the
order of two equal elements is maintained).
If a key function is given, apply it once to each list item and sort them,
ascending or descending, according to their function values.
The reverse flag can be set to sort in descending order.
<h3 id="__add__"><code>__add__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    value, /
)
</code></pre>
Return self+value.

<h3 id="__contains__"><code>__contains__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    key, /
)
</code></pre>
Return key in self.

<h3 id="__eq__"><code>__eq__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    value, /
)
</code></pre>
Return self==value.

<h3 id="__ge__"><code>__ge__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    value, /
)
</code></pre>
Return self>=value.

<h3 id="__getitem__"><code>__getitem__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__()
</code></pre>
x.__getitem__(y) <==> x[y]

<h3 id="__gt__"><code>__gt__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    value, /
)
</code></pre>
Return self>value.

<h3 id="__le__"><code>__le__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    value, /
)
</code></pre>
Return self<=value.

<h3 id="__lt__"><code>__lt__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    value, /
)
</code></pre>
Return self<value.

<h3 id="__mul__"><code>__mul__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    value, /
)
</code></pre>
Return self*value.

<h3 id="__ne__"><code>__ne__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    value, /
)
</code></pre>
Return self!=value.

<h3 id="__rmul__"><code>__rmul__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    value, /
)
</code></pre>
Return value*self.


