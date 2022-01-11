# TFSimilarity.indexer.defaultdict






defaultdict(default_factory[, ...]) --> dict with default factory

```python
TFSimilarity.indexer.defaultdict(
    *args, **kwargs
)
```



<!-- Placeholder for "Used in" -->

The default factory is called without arguments to produce
a new value when a key is not present, in __getitem__ only.
A defaultdict compares equal to a dict with the same items.
All remaining arguments are treated the same as if they were
passed to the dict constructor, including keyword arguments.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>default_factory</b>
</td>
<td>
Factory for default value called by __missing__().
</td>
</tr>
</table>



## Methods

<h3 id="clear">clear</h3>

```python
clear()
```


D.clear() -> None.  Remove all items from D.


<h3 id="copy">copy</h3>

```python
copy()
```


D.copy() -> a shallow copy of D.


<h3 id="fromkeys">fromkeys</h3>

```python
fromkeys(
    value, /
)
```


Create a new dictionary with keys from iterable and values set to value.


<h3 id="get">get</h3>

```python
get(
    key, default, /
)
```


Return the value for key if key is in the dictionary, else default.


<h3 id="items">items</h3>

```python
items()
```


D.items() -> a set-like object providing a view on D's items


<h3 id="keys">keys</h3>

```python
keys()
```


D.keys() -> a set-like object providing a view on D's keys


<h3 id="pop">pop</h3>

```python
pop()
```


D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised

<h3 id="popitem">popitem</h3>

```python
popitem()
```


D.popitem() -> (k, v), remove and return some (key, value) pair as a
2-tuple; but raise KeyError if D is empty.

<h3 id="setdefault">setdefault</h3>

```python
setdefault(
    key, default, /
)
```


Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.

<h3 id="update">update</h3>

```python
update()
```


D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]

<h3 id="values">values</h3>

```python
values()
```


D.values() -> an object providing a view on D's values


<h3 id="__contains__">__contains__</h3>

```python
__contains__(
    key, /
)
```


True if the dictionary has the specified key, else False.


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
__getitem__()
```


x.__getitem__(y) <==> x[y]


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


<h3 id="__ne__">__ne__</h3>

```python
__ne__(
    value, /
)
```


Return self!=value.




