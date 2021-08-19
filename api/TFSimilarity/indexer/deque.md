
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.indexer.deque" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="appendleft"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="count"/>
<meta itemprop="property" content="extend"/>
<meta itemprop="property" content="extendleft"/>
<meta itemprop="property" content="index"/>
<meta itemprop="property" content="insert"/>
<meta itemprop="property" content="pop"/>
<meta itemprop="property" content="popleft"/>
<meta itemprop="property" content="remove"/>
<meta itemprop="property" content="reverse"/>
<meta itemprop="property" content="rotate"/>
<meta itemprop="property" content="maxlen"/>
</div>
# TFSimilarity.indexer.deque
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
</table>

deque([iterable[, maxlen]]) --> deque object
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.indexer.deque(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->
A list-like sequence optimized for data accesses near its endpoints.
## Methods
<h3 id="append"><code>append</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>append()
</code></pre>
Add an element to the right side of the deque.

<h3 id="appendleft"><code>appendleft</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>appendleft()
</code></pre>
Add an element to the left side of the deque.

<h3 id="clear"><code>clear</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>
Remove all elements from the deque.

<h3 id="copy"><code>copy</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy()
</code></pre>
Return a shallow copy of a deque.

<h3 id="count"><code>count</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>count()
</code></pre>
D.count(value) -> integer -- return number of occurrences of value

<h3 id="extend"><code>extend</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>extend()
</code></pre>
Extend the right side of the deque with elements from the iterable

<h3 id="extendleft"><code>extendleft</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>extendleft()
</code></pre>
Extend the left side of the deque with elements from the iterable

<h3 id="index"><code>index</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>index()
</code></pre>
D.index(value, [start, [stop]]) -> integer -- return first index of value.
Raises ValueError if the value is not present.
<h3 id="insert"><code>insert</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>insert()
</code></pre>
D.insert(index, object) -- insert object before index

<h3 id="pop"><code>pop</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pop()
</code></pre>
Remove and return the rightmost element.

<h3 id="popleft"><code>popleft</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>popleft()
</code></pre>
Remove and return the leftmost element.

<h3 id="remove"><code>remove</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove()
</code></pre>
D.remove(value) -- remove first occurrence of value.

<h3 id="reverse"><code>reverse</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reverse()
</code></pre>
D.reverse() -- reverse *IN PLACE*

<h3 id="rotate"><code>rotate</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>rotate()
</code></pre>
Rotate the deque n steps to the right (default n=1).  If n is negative, rotates left.

<h3 id="__add__"><code>__add__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    value, /
)
</code></pre>
Return self+value.

<h3 id="__bool__"><code>__bool__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>
self != 0

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
<code>__getitem__(
    key, /
)
</code></pre>
Return self[key].

<h3 id="__gt__"><code>__gt__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    value, /
)
</code></pre>
Return self>value.

<h3 id="__iter__"><code>__iter__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>
Implement iter(self).

<h3 id="__le__"><code>__le__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    value, /
)
</code></pre>
Return self<=value.

<h3 id="__len__"><code>__len__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>
Return len(self).

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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>
<tr>
<td>
maxlen<a id="maxlen"></a>
</td>
<td>
Instance of `getset_descriptor`
maximum size of a deque or None if unbounded
</td>
</tr>
</table>
