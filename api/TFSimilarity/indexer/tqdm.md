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

<h3 id="as_completed"><code>as_completed</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>as_completed(
    fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs
)
</code></pre>

Wrapper for <b>asyncio.as_completed</b>.


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

Use <b>self.sp</b> to display <b>msg</b> in the specified <b>pos</b>.

Consider overloading this function when inheriting to use e.g.:
<b>self.some_frontend(**self.format_dict)</b> instead of <b>self.sp</b>.

Parameters
----------
msg  : str, optional. What to display (default: <b>repr(self)</b>).
pos  : int, optional. Position to <b>moveto</b>
  (default: <b>abs(self.pos)</b>).

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
    num, suffix=&#x27;&## # Now you can use <b>progress_apply</b> instead of <b>apply</b>
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
    If <b>True</b>, does not lock.
    If [default: <b>False</b>]: calls <b>acquire()</b> on internal lock.
lock_args  : tuple, optional
    Passed to internal lock's <b>acquire()</b>.
    If specified, will only <b>display()</b> if <b>acquire()</b> returns <b>True</b>.

<h3 id="reset"><code>reset</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset(
    total=None
)
</code></pre>

Resets to 0 iterations for repeated use.

Consider combining with <b>leave=True</b>.

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
    s=&#x27;&## Initialise
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

<h3 id="wrapattr"><code>wrapattr</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>wrapattr(
    stream, method, total=None, bytes=(True), **tqdm_kwargs
)
</code></pre>

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

