# TFSimilarity.indexer.deque






deque([iterable[, maxlen]]) --> deque object

```python
TFSimilarity.indexer.deque(
    *args, **kwargs
)
```



<!-- Placeholder for "Used in" -->

A list-like sequence optimized for data accesses near its endpoints.

## Methods

<h3 id="append">append</h3>

```python
append()
```


Add an element to the right side of the deque.


<h3 id="appendleft">appendleft</h3>

```python
appendleft()
```


Add an element to the left side of the deque.


<h3 id="clear">clear</h3>

```python
clear()
```


Remove all elements from the deque.


<h3 id="copy">copy</h3>

```python
copy()
```


Return a shallow copy of a deque.


<h3 id="count">count</h3>

```python
count()
```


D.count(value) -> integer -- return number of occurrences of value


<h3 id="extend">extend</h3>

```python
extend()
```


Extend the right side of the deque with elements from the iterable


<h3 id="extendleft">extendleft</h3>

```python
extendleft()
```


Extend the left side of the deque with elements from the iterable


<h3 id="index">index</h3>

```python
index()
```


D.index(value, [start, [stop]]) -> integer -- return first index of value.
Raises ValueError if the value is not present.

<h3 id="insert">insert</h3>

```python
insert()
```


D.insert(index, object) -- insert object before index


<h3 id="pop">pop</h3>

```python
pop()
```


Remove and return the rightmost element.


<h3 id="popleft">popleft</h3>

```python
popleft()
```


Remove and return the leftmost element.


<h3 id="remove">remove</h3>

```python
remove()
```


D.remove(value) -- remove first occurrence of value.


<h3 id="reverse">reverse</h3>

```python
reverse()
```


D.reverse() -- reverse *IN PLACE*


<h3 id="rotate">rotate</h3>

```python
rotate()
```


Rotate the deque n steps to the right (default n=1).  If n is negative, rotates left.


<h3 id="__add__">__add__</h3>

```python
__add__(
    value, /
)
```


Return self+value.


<h3 id="__bool__">__bool__</h3>

```python
__bool__()
```


self != 0


<h3 id="__contains__">__contains__</h3>

```python
__contains__(
    key, /
)
```


Return key in self.


<h3 id="__eq__">__eq__</h3>

```python
__eq__(
    value, /
)
```


Return self==value.


<h3 id="__ge__">__ge__</h3>

```python
__ge__(
    value, /
)
```


Return self>=value.


<h3 id="__getitem__">__getitem__</h3>

```python
__getitem__(
    key, /
)
```


Return self[key].


<h3 id="__gt__">__gt__</h3>

```python
__gt__(
    value, /
)
```


Return self>value.


<h3 id="__iter__">__iter__</h3>

```python
__iter__()
```


Implement iter(self).


<h3 id="__le__">__le__</h3>

```python
__le__(
    value, /
)
```


Return self<=value.


<h3 id="__len__">__len__</h3>

```python
__len__()
```


Return len(self).


<h3 id="__lt__">__lt__</h3>

```python
__lt__(
    value, /
)
```


Return self<value.


<h3 id="__mul__">__mul__</h3>

```python
__mul__(
    value, /
)
```


Return self*value.


<h3 id="__ne__">__ne__</h3>

```python
__ne__(
    value, /
)
```


Return self!=value.


<h3 id="__rmul__">__rmul__</h3>

```python
__rmul__(
    value, /
)
```


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
Instance of <b>getset_descriptor</b>

maximum size of a deque or None if unbounded
</td>
</tr>
</table>

