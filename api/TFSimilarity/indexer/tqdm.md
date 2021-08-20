# TFSimilarity.indexer.tqdm






Asynchronous-friendly version of tqdm (Python 3.6+).

```python
TFSimilarity.indexer.tqdm(
    iterable=None, *args, **kwargs
)
```



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>format_dict</b>
</td>
<td>
Public API for read-only member access.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
<b>format_dict</b>
</td>
<td>
Public API for read-only member access.
</td>
</tr>
</table>



## Methods

<h3 id="as_completed">as_completed</h3>

``<b>python
@classmethod</b>``

```python
as_completed(
    fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs
)
```


Wrapper for <b>asyncio.as_completed</b>.


<h3 id="clear">clear</h3>

```python
clear(
    nolock=(False)
)
```


Clear current bar display.


<h3 id="close">close</h3>

```python
close()
```


Cleanup and (if leave=False) close the progressbar.


<h3 id="display">display</h3>

```python
display(
    msg=None, pos=None
)
```


Use <b>self.sp</b> to display <b>msg</b> in the specified <b>pos</b>.

Consider overloading this function when inheriting to use e.g.:
<b>self.some_frontend(**self.format_dict)</b> instead of <b>self.sp</b>.

Parameters
----------
msg  : str, optional. What to display (default: <b>repr(self)</b>).
pos  : int, optional. Position to <b>moveto</b>
  (default: <b>abs(self.pos)</b>).

<h3 id="external_write_mode">external_write_mode</h3>

``<b>python
@classmethod</b>``

```python
external_write_mode(
    file=None, nolock=(False)
)
```


Disable tqdm within context and refresh tqdm when exits.
Useful when writing to standard output stream

<h3 id="format_interval">format_interval</h3>

``<b>python
@staticmethod</b>``

```python
format_interval(
    t
)
```


Formats a number of seconds as a clock time, [H:]MM:SS

Parameters
----------
t  : int
    Number of seconds.

Returns
-------
out  : str
    [H:]MM:SS

<h3 id="format_meter">format_meter</h3>

``<b>python
@staticmethod</b>``

```python
format_meter(
    n, total, elapsed, ncols=None, prefix=, ascii=(False),
    unit=it, unit_scale=(False), rate=None, bar_format=None,
    postfix=None, unit_divisor=1000, initial=0, colour=None, **extra_kwargs
)
```


Return a string-based progress bar given some parameters

Parameters
----------
n  : int or float
    Number of finished iterations.
total  : int or float
    The expected total number of iterations. If meaningless (None),
    only basic progress statistics are displayed (no ETA).
elapsed  : float
    Number of seconds passed since start.
ncols  : int, optional
    The width of the entire output message. If specified,
    dynamically resizes <b><i>bar</i></b> to stay within this bound
    [default: None]. If <b>0</b>, will not print any bar (only stats).
    The fallback is <b><i>bar:10</i></b>.
prefix  : str, optional
    Prefix message (included in total width) [default: ''].
    Use as <i>desc</i> in bar_format string.
ascii  : bool, optional or str, optional
    If not set, use unicode (smooth blocks) to fill the meter
    [default: False]. The fallback is to use ASCII characters
    " 123456789#".
unit  : str, optional
    The iteration unit [default: 'it'].
unit_scale  : bool or int or float, optional
    If 1 or True, the number of iterations will be printed with an
    appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
    [default: False]. If any other non-zero number, will scale
    <b>total</b> and <b>n</b>.
rate  : float, optional
    Manual override for iteration rate.
    If [default: None], uses n/elapsed.
bar_format  : str, optional
    Specify a custom bar string formatting. May impact performance.
    [default: '<i>l_bar}{bar}{r_bar</i>'], where
    l_bar='<i>desc}: {percentage:3.0f</i>%|' and
    r_bar='| <i>n_fmt}/{total_fmt} [{elapsed}<{remaining</i>, '
      '<i>rate_fmt}{postfix</i>]'
    Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
      percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
      rate, rate_fmt, rate_noinv, rate_noinv_fmt,
      rate_inv, rate_inv_fmt, postfix, unit_divisor,
      remaining, remaining_s, eta.
    Note that a trailing ": " is automatically removed after <i>desc</i>
    if the latter is empty.
postfix  : *, optional
    Similar to <b>prefix</b>, but placed at the end
    (e.g. for additional stats).
    Note: postfix is usually a string (not a dict) for this method,
    and will if possible be set to postfix = ', ' + postfix.
    However other types are supported (#382).
unit_divisor  : float, optional
    [default: 1000], ignored unless <b>unit_scale</b> is True.
initial  : int or float, optional
    The initial counter value [default: 0].
colour  : str, optional
    Bar colour (e.g. 'green', '#00ff00').

Returns
-------
out  : Formatted meter and stats, ready to display.

<h3 id="format_num">format_num</h3>

``<b>python
@staticmethod</b>``

```python
format_num(
    n
)
```


Intelligent scientific notation (.3g).

Parameters
----------
n  : int or float or Numeric
    A Number.

Returns
-------
out  : str
    Formatted number.

<h3 id="format_sizeof">format_sizeof</h3>

``<b>python
@staticmethod</b>``

```python
format_sizeof(
    num, suffix=&## # Now you can use <b>progress_apply</b> instead of <b>apply</b>
>>> df.groupby(0).progress_apply(lambda x: x**2)
```

References
----------
<https://stackoverflow.com/questions/18603270/        progress-indicator-during-pandas-operations-python>

<h3 id="refresh">refresh</h3>

```python
refresh(
    nolock=(False), lock_args=None
)
```


Force refresh the display of this bar.

Parameters
----------
nolock  : bool, optional
    If <b>True</b>, does not lock.
    If [default: <b>False</b>]: calls <b>acquire()</b> on internal lock.
lock_args  : tuple, optional
    Passed to internal lock's <b>acquire()</b>.
    If specified, will only <b>display()</b> if <b>acquire()</b> returns <b>True</b>.

<h3 id="reset">reset</h3>

```python
reset(
    total=None
)
```


Resets to 0 iterations for repeated use.

Consider combining with <b>leave=True</b>.

Parameters
----------
total  : int or float, optional. Total to use for the new bar.

<h3 id="send">send</h3>

```python
send(
    *args, **kwargs
)
```





<h3 id="set_description">set_description</h3>

```python
set_description(
    desc=None, refresh=(True)
)
```


Set/modify description of the progress bar.

Parameters
----------
desc  : str, optional
refresh  : bool, optional
    Forces refresh [default: True].

<h3 id="set_description_str">set_description_str</h3>

```python
set_description_str(
    desc=None, refresh=(True)
)
```


Set/modify description without ': ' appended.


<h3 id="set_lock">set_lock</h3>

``<b>python
@classmethod</b>``

```python
set_lock(
    lock
)
```


Set the global lock.


<h3 id="set_postfix">set_postfix</h3>

```python
set_postfix(
    ordered_dict=None, refresh=(True), **kwargs
)
```


Set/modify postfix (additional stats)
with automatic formatting based on datatype.

Parameters
----------
ordered_dict  : dict or OrderedDict, optional
refresh  : bool, optional
    Forces refresh [default: True].
kwargs  : dict, optional

<h3 id="set_postfix_str">set_postfix_str</h3>

```python
set_postfix_str(
    s=&## Initialise
>>> for current_buffer in stream:
...    ...
...    t.update(len(current_buffer))
>>> t.close()
The last line is highly recommended, but possibly not necessary if
<b>t.update()</b> will be called in such a way that <b>filesize</b> will be
exactly reached and printed.
```

Parameters
----------
n  : int or float, optional
    Increment to add to the internal counter of iterations
    [default: 1]. If using float, consider specifying <b><i>n:.3f</i></b>
    or similar in <b>bar_format</b>, or specifying <b>unit_scale</b>.

Returns
-------
out  : bool or None
    True if a <b>display()</b> was triggered.

<h3 id="wrapattr">wrapattr</h3>

``<b>python
@classmethod</b>``

```python
wrapattr(
    stream, method, total=None, bytes=(True), **tqdm_kwargs
)
```


stream  : file-like object.
method  : str, "read" or "write". The result of <b>read()</b> and
    the first argument of <b>write()</b> should have a <b>len()</b>.

```
>>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:
...     while True:
...         chunk = fobj.read(chunk_size)
...         if not chunk:
...             break
```

<h3 id="write">write</h3>

``<b>python
@classmethod</b>``

```python
write(
    s, file=None, end=\n, nolock=(False)
)
```


Print a message via tqdm (without overlap with bars).


<h3 id="__bool__">__bool__</h3>

```python
__bool__()
```





<h3 id="__enter__">__enter__</h3>

```python
__enter__()
```





<h3 id="__eq__">__eq__</h3>

```python
__eq__(
    other
)
```


Return self==value.


<h3 id="__exit__">__exit__</h3>

```python
__exit__(
    exc_type, exc_value, traceback
)
```





<h3 id="__ge__">__ge__</h3>

```python
__ge__(
    other
)
```


Return self>=value.


<h3 id="__gt__">__gt__</h3>

```python
__gt__(
    other
)
```


Return self>value.


<h3 id="__iter__">__iter__</h3>

```python
__iter__()
```


Backward-compatibility to use: for x in tqdm(iterable)


<h3 id="__le__">__le__</h3>

```python
__le__(
    other
)
```


Return self<=value.


<h3 id="__len__">__len__</h3>

```python
__len__()
```





<h3 id="__lt__">__lt__</h3>

```python
__lt__(
    other
)
```


Return self<value.


<h3 id="__ne__">__ne__</h3>

```python
__ne__(
    other
)
```


Return self!=value.


<h3 id="__nonzero__">__nonzero__</h3>

```python
__nonzero__()
```









<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
monitor<a id="monitor"></a>
</td>
<td>
<b>None</b>
</td>
</tr><tr>
<td>
monitor_interval<a id="monitor_interval"></a>
</td>
<td>
<b>10</b>
</td>
</tr>
</table>

