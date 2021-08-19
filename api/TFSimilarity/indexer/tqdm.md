
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="TFSimilarity.indexer.tqdm" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="__nonzero__"/>
<meta itemprop="property" content="as_completed"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="close"/>
<meta itemprop="property" content="display"/>
<meta itemprop="property" content="external_write_mode"/>
<meta itemprop="property" content="format_interval"/>
<meta itemprop="property" content="format_meter"/>
<meta itemprop="property" content="format_num"/>
<meta itemprop="property" content="format_sizeof"/>
<meta itemprop="property" content="gather"/>
<meta itemprop="property" content="get_lock"/>
<meta itemprop="property" content="moveto"/>
<meta itemprop="property" content="pandas"/>
<meta itemprop="property" content="refresh"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="send"/>
<meta itemprop="property" content="set_description"/>
<meta itemprop="property" content="set_description_str"/>
<meta itemprop="property" content="set_lock"/>
<meta itemprop="property" content="set_postfix"/>
<meta itemprop="property" content="set_postfix_str"/>
<meta itemprop="property" content="status_printer"/>
<meta itemprop="property" content="unpause"/>
<meta itemprop="property" content="update"/>
<meta itemprop="property" content="wrapattr"/>
<meta itemprop="property" content="write"/>
<meta itemprop="property" content="monitor"/>
<meta itemprop="property" content="monitor_interval"/>
</div>
# TFSimilarity.indexer.tqdm
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
</table>

Asynchronous-friendly version of tqdm (Python 3.6+).
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.indexer.tqdm(
    iterable=None, *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`format_dict`
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
`format_dict`
</td>
<td>
Public API for read-only member access.
</td>
</tr>
</table>

## Methods
<h3 id="as_completed"><code>as_completed</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>as_completed(
    fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs
)
</code></pre>
Wrapper for `asyncio.as_completed`.

<h3 id="clear"><code>clear</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear(
    nolock=(False)
)
</code></pre>
Clear current bar display.

<h3 id="close"><code>close</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>close()
</code></pre>
Cleanup and (if leave=False) close the progressbar.

<h3 id="display"><code>display</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>display(
    msg=None, pos=None
)
</code></pre>
Use `self.sp` to display `msg` in the specified `pos`.
Consider overloading this function when inheriting to use e.g.:
`self.some_frontend(**self.format_dict)` instead of `self.sp`.
Parameters
----------
msg  : str, optional. What to display (default: `repr(self)`).
pos  : int, optional. Position to `moveto`
  (default: `abs(self.pos)`).
<h3 id="external_write_mode"><code>external_write_mode</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>external_write_mode(
    file=None, nolock=(False)
)
</code></pre>
Disable tqdm within context and refresh tqdm when exits.
Useful when writing to standard output stream
<h3 id="format_interval"><code>format_interval</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>format_interval(
    t
)
</code></pre>
Formats a number of seconds as a clock time, [H:]MM:SS
Parameters
----------
t  : int
    Number of seconds.
Returns
-------
out  : str
    [H:]MM:SS
<h3 id="format_meter"><code>format_meter</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>format_meter(
    n, total, elapsed, ncols=None, prefix=&#x27;&#x27;, ascii=(False),
    unit=&#x27;it&#x27;, unit_scale=(False), rate=None, bar_format=None,
    postfix=None, unit_divisor=1000, initial=0, colour=None, **extra_kwargs
)
</code></pre>
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
    dynamically resizes `{bar}` to stay within this bound
    [default: None]. If `0`, will not print any bar (only stats).
    The fallback is `{bar:10}`.
prefix  : str, optional
    Prefix message (included in total width) [default: ''].
    Use as {desc} in bar_format string.
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
    `total` and `n`.
rate  : float, optional
    Manual override for iteration rate.
    If [default: None], uses n/elapsed.
bar_format  : str, optional
    Specify a custom bar string formatting. May impact performance.
    [default: '{l_bar}{bar}{r_bar}'], where
    l_bar='{desc}: {percentage:3.0f}%|' and
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
      '{rate_fmt}{postfix}]'
    Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
      percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
      rate, rate_fmt, rate_noinv, rate_noinv_fmt,
      rate_inv, rate_inv_fmt, postfix, unit_divisor,
      remaining, remaining_s, eta.
    Note that a trailing ": " is automatically removed after {desc}
    if the latter is empty.
postfix  : *, optional
    Similar to `prefix`, but placed at the end
    (e.g. for additional stats).
    Note: postfix is usually a string (not a dict) for this method,
    and will if possible be set to postfix = ', ' + postfix.
    However other types are supported (#382).
unit_divisor  : float, optional
    [default: 1000], ignored unless `unit_scale` is True.
initial  : int or float, optional
    The initial counter value [default: 0].
colour  : str, optional
    Bar colour (e.g. 'green', '#00ff00').
Returns
-------
out  : Formatted meter and stats, ready to display.
<h3 id="format_num"><code>format_num</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>format_num(
    n
)
</code></pre>
Intelligent scientific notation (.3g).
Parameters
----------
n  : int or float or Numeric
    A Number.
Returns
-------
out  : str
    Formatted number.
<h3 id="format_sizeof"><code>format_sizeof</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>format_sizeof(
    num, suffix=&#x27;&#x27;, divisor=1000
)
</code></pre>
Formats a number (greater than unity) with SI Order of Magnitude
prefixes.
Parameters
----------
num  : float
    Number ( >= 1) to format.
suffix  : str, optional
    Post-postfix [default: ''].
divisor  : float, optional
    Divisor between prefixes [default: 1000].
Returns
-------
out  : str
    Number with Order of Magnitude SI unit postfix.
<h3 id="gather"><code>gather</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gather(
    *fs, loop=None, timeout=None, total=None, **tqdm_kwargs
)
</code></pre>
Wrapper for `asyncio.gather`.

<h3 id="get_lock"><code>get_lock</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>get_lock()
</code></pre>
Get the global lock. Construct it if it does not exist.

<h3 id="moveto"><code>moveto</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>moveto(
    n
)
</code></pre>


<h3 id="pandas"><code>pandas</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>pandas(
    **tqdm_kwargs
)
</code></pre>
Registers the current `tqdm` class with
    pandas.core.
    ( frame.DataFrame
    | series.Series
    | groupby.(generic.)DataFrameGroupBy
    | groupby.(generic.)SeriesGroupBy
    ).progress_apply
A new instance will be create every time `progress_apply` is called,
and each instance will automatically `close()` upon completion.
Parameters
----------
tqdm_kwargs  : arguments for the tqdm instance
Examples
--------
```
>>> import pandas as pd
>>> import numpy as np
>>> from tqdm import tqdm
>>> from tqdm.gui import tqdm as tqdm_gui
>>>
>>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
>>> tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
>>> # Now you can use `progress_apply` instead of `apply`
>>> df.groupby(0).progress_apply(lambda x: x**2)
```
References
----------
<https://stackoverflow.com/questions/18603270/        progress-indicator-during-pandas-operations-python>
<h3 id="refresh"><code>refresh</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>refresh(
    nolock=(False), lock_args=None
)
</code></pre>
Force refresh the display of this bar.
Parameters
----------
nolock  : bool, optional
    If `True`, does not lock.
    If [default: `False`]: calls `acquire()` on internal lock.
lock_args  : tuple, optional
    Passed to internal lock's `acquire()`.
    If specified, will only `display()` if `acquire()` returns `True`.
<h3 id="reset"><code>reset</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset(
    total=None
)
</code></pre>
Resets to 0 iterations for repeated use.
Consider combining with `leave=True`.
Parameters
----------
total  : int or float, optional. Total to use for the new bar.
<h3 id="send"><code>send</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>send(
    *args, **kwargs
)
</code></pre>


<h3 id="set_description"><code>set_description</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_description(
    desc=None, refresh=(True)
)
</code></pre>
Set/modify description of the progress bar.
Parameters
----------
desc  : str, optional
refresh  : bool, optional
    Forces refresh [default: True].
<h3 id="set_description_str"><code>set_description_str</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_description_str(
    desc=None, refresh=(True)
)
</code></pre>
Set/modify description without ': ' appended.

<h3 id="set_lock"><code>set_lock</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>set_lock(
    lock
)
</code></pre>
Set the global lock.

<h3 id="set_postfix"><code>set_postfix</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_postfix(
    ordered_dict=None, refresh=(True), **kwargs
)
</code></pre>
Set/modify postfix (additional stats)
with automatic formatting based on datatype.
Parameters
----------
ordered_dict  : dict or OrderedDict, optional
refresh  : bool, optional
    Forces refresh [default: True].
kwargs  : dict, optional
<h3 id="set_postfix_str"><code>set_postfix_str</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_postfix_str(
    s=&#x27;&#x27;, refresh=(True)
)
</code></pre>
Postfix without dictionary expansion, similar to prefix handling.

<h3 id="status_printer"><code>status_printer</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>status_printer(
    file
)
</code></pre>
Manage the printing and in-place updating of a line of characters.
Note that if the string is longer than a line, then in-place
updating may not work (it will print a new line at each refresh).
<h3 id="unpause"><code>unpause</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unpause()
</code></pre>
Restart tqdm timer from last print time.

<h3 id="update"><code>update</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update(
    n=1
)
</code></pre>
Manually update the progress bar, useful for streams
such as reading files.
E.g.:
```
>>> t = tqdm(total=filesize) # Initialise
>>> for current_buffer in stream:
...    ...
...    t.update(len(current_buffer))
>>> t.close()
The last line is highly recommended, but possibly not necessary if
`t.update()` will be called in such a way that `filesize` will be
exactly reached and printed.
```
Parameters
----------
n  : int or float, optional
    Increment to add to the internal counter of iterations
    [default: 1]. If using float, consider specifying `{n:.3f}`
    or similar in `bar_format`, or specifying `unit_scale`.
Returns
-------
out  : bool or None
    True if a `display()` was triggered.
<h3 id="wrapattr"><code>wrapattr</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>wrapattr(
    stream, method, total=None, bytes=(True), **tqdm_kwargs
)
</code></pre>
stream  : file-like object.
method  : str, "read" or "write". The result of `read()` and
    the first argument of `write()` should have a `len()`.
```
>>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:
...     while True:
...         chunk = fobj.read(chunk_size)
...         if not chunk:
...             break
```
<h3 id="write"><code>write</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>write(
    s, file=None, end=&#x27;\n&#x27;, nolock=(False)
)
</code></pre>
Print a message via tqdm (without overlap with bars).

<h3 id="__bool__"><code>__bool__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>


<h3 id="__enter__"><code>__enter__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__enter__()
</code></pre>


<h3 id="__eq__"><code>__eq__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>
Return self==value.

<h3 id="__exit__"><code>__exit__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__exit__(
    exc_type, exc_value, traceback
)
</code></pre>


<h3 id="__ge__"><code>__ge__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other
)
</code></pre>
Return self>=value.

<h3 id="__gt__"><code>__gt__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other
)
</code></pre>
Return self>value.

<h3 id="__iter__"><code>__iter__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>
Backward-compatibility to use: for x in tqdm(iterable)

<h3 id="__le__"><code>__le__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other
)
</code></pre>
Return self<=value.

<h3 id="__len__"><code>__len__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>


<h3 id="__lt__"><code>__lt__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>
Return self<value.

<h3 id="__ne__"><code>__ne__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>
Return self!=value.

<h3 id="__nonzero__"><code>__nonzero__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__nonzero__()
</code></pre>




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>
<tr>
<td>
monitor<a id="monitor"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
monitor_interval<a id="monitor_interval"></a>
</td>
<td>
`10`
</td>
</tr>
</table>
