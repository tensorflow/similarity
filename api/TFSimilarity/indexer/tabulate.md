# TFSimilarity.indexer.tabulate






Format a fixed width table for pretty printing.

```python
TFSimilarity.indexer.tabulate(
    tabular_data, headers=(), tablefmt=simple,
    floatfmt=_DEFAULT_FLOATFMT, numalign=_DEFAULT_ALIGN, stralign=_DEFAULT_ALIGN,
    missingval=_DEFAULT_MISSINGVAL, showindex=default,
    disable_numparse=(False), colalign=None
)
```



<!-- Placeholder for "Used in" -->

```
>>> print(tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]]))
---  ---------
  1      2.34
-56      8.999
  2  10001
---  ---------
```

The first required argument (<b>tabular_data</b>) can be a
list-of-lists (or another iterable of iterables), a list of named
tuples, a dictionary of iterables, an iterable of dictionaries,
a two-dimensional NumPy array, NumPy record array, or a Pandas'
dataframe.


Table headers
-------------

To print nice column headers, supply the second argument (<b>headers</b>):

  - <b>headers</b> can be an explicit list of column headers
  - if <b>headers="firstrow"</b>, then the first row of data is used
  - if <b>headers="keys"</b>, then dictionary keys or column indices are used

Otherwise a headerless table is produced.

If the number of headers is less than the number of columns, they
are supposed to be names of the last columns. This is consistent
with the plain-text format of R and Pandas' dataframes.

```
>>> print(tabulate([["sex","age"],["Alice","F",24],["Bob","M",19]],
...       headers="firstrow"))
       sex      age
-----  -----  -----
Alice  F         24
Bob    M         19
```

By default, pandas.DataFrame data have an additional column called
row index. To add a similar column to all other types of data,
use <b>showindex="always"</b> or <b>showindex=True</b>. To suppress row indices
for all types of data, pass <b>showindex="never" or </b>showindex=False`.
To add a custom row index column, pass <b>showindex=some_iterable</b>.

```
>>> print(tabulate([["F",24],["M",19]], showindex="always"))
-  -  --
0  F  24
1  M  19
-  -  --
```


Column alignment
----------------

<b>tabulate</b> tries to detect column types automatically, and aligns
the values properly. By default it aligns decimal points of the
numbers (or flushes integer numbers to the right), and flushes
everything else to the left. Possible column alignments
(<b>numalign</b>, <b>stralign</b>) are: "right", "center", "left", "decimal"
(only for <b>numalign</b>), and None (to disable alignment).


Table formats
-------------

<b>floatfmt</b> is a format specification used for columns which
contain numeric data with a decimal point. This can also be
a list or tuple of format strings, one per column.

<b>None</b> values are replaced with a <b>missingval</b> string (like
<b>floatfmt</b>, this can also be a list of values for different
columns):

```
>>> print(tabulate([["spam", 1, None],
...                 ["eggs", 42, 3.14],
...                 ["other", None, 2.7]], missingval="?"))
-----  --  ----
spam    1  ?
eggs   42  3.14
other   ?  2.7
-----  --  ----
```

Various plain-text table formats (<b>tablefmt</b>) are supported:
'plain', 'simple', 'grid', 'pipe', 'orgtbl', 'rst', 'mediawiki',
'latex', 'latex_raw', 'latex_booktabs', 'latex_longtable' and tsv.
Variable <b>tabulate_formats</b>contains the list of currently supported formats.

"plain" format doesn't use any pseudographics to draw tables,
it separates columns with a double space:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                 ["strings", "numbers"], "plain"))
strings      numbers
spam         41.9999
eggs        451
```

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="plain"))
spam   41.9999
eggs  451
```

"simple" format is like Pandoc simple_tables:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                 ["strings", "numbers"], "simple"))
strings      numbers
---------  ---------
spam         41.9999
eggs        451
```

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="simple"))
----  --------
spam   41.9999
eggs  451
----  --------
```

"grid" is similar to tables produced by Emacs table.el package or
Pandoc grid_tables:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                ["strings", "numbers"], "grid"))
+-----------+-----------+
| strings   |   numbers |
+===========+===========+
| spam      |   41.9999 |
+-----------+-----------+
| eggs      |  451      |
+-----------+-----------+
```

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="grid"))
+------+----------+
| spam |  41.9999 |
+------+----------+
| eggs | 451      |
+------+----------+
```

"fancy_grid" draws a grid using box-drawing characters:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                ["strings", "numbers"], "fancy_grid"))
╒═══════════╤═══════════╕
│ strings   │   numbers │
╞═══════════╪═══════════╡
│ spam      │   41.9999 │
├───────────┼───────────┤
│ eggs      │  451      │
╘═══════════╧═══════════╛
```

"pipe" is like tables in PHP Markdown Extra extension or Pandoc
pipe_tables:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                ["strings", "numbers"], "pipe"))
| strings   |   numbers |
|:----------|----------:|
| spam      |   41.9999 |
| eggs      |  451      |
```

"presto" is like tables produce by the Presto CLI:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                ["strings", "numbers"], "presto"))
 strings   |   numbers
-----------+-----------
 spam      |   41.9999
 eggs      |  451
```

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="pipe"))
|:-----|---------:|
| spam |  41.9999 |
| eggs | 451      |
```

"orgtbl" is like tables in Emacs org-mode and orgtbl-mode. They
are slightly different from "pipe" format by not using colons to
define column alignment, and using a "+" sign to indicate line
intersections:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                ["strings", "numbers"], "orgtbl"))
| strings   |   numbers |
|-----------+-----------|
| spam      |   41.9999 |
| eggs      |  451      |
```


```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="orgtbl"))
| spam |  41.9999 |
| eggs | 451      |
```

"rst" is like a simple table format from reStructuredText; please
note that reStructuredText accepts also "grid" tables:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
...                ["strings", "numbers"], "rst"))
=========  =========
strings      numbers
=========  =========
spam         41.9999
eggs        451
=========  =========
```

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="rst"))
====  ========
spam   41.9999
eggs  451
====  ========
```

"mediawiki" produces a table markup used in Wikipedia and on other
MediaWiki-based sites:

```
>>> print(tabulate([["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]],
...                headers="firstrow", tablefmt="mediawiki"))
{| class="wikitable" style="text-align: left;"
|+ <!-- caption -->
|-
! strings   !! align="right"|   numbers
|-
| spam      || align="right"|   41.9999
|-
| eggs      || align="right"|  451
|}
```

"html" produces HTML markup as an html.escape'd str
with a ._repr_html_ method so that Jupyter Lab and Notebook display the HTML
and a .str property so that the raw HTML remains accessible
the unsafehtml table format can be used if an unescaped HTML format is required:

```
>>> print(tabulate([["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]],
...                headers="firstrow", tablefmt="html"))
<table>
<thead>
<tr><th>strings  </th><th style="text-align: right;">  numbers</th></tr>
</thead>
<tbody>
<tr><td>spam     </td><td style="text-align: right;">  41.9999</td></tr>
<tr><td>eggs     </td><td style="text-align: right;"> 451     </td></tr>
</tbody>
</table>
```

"latex" produces a tabular environment of LaTeX document markup:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="latex"))
\begin<i>tabular}{lr</i>
\hline
 spam &  41.9999 \\
 eggs & 451      \\
\hline
\end<i>tabular</i>
```

"latex_raw" is similar to "latex", but doesn't escape special characters,
such as backslash and underscore, so LaTeX commands may embedded into
cells' values:

```
>>> print(tabulate([["spam$_9$", 41.9999], ["\\emph<i>eggs</i>", "451.0"]], tablefmt="latex_raw"))
\begin<i>tabular}{lr</i>
\hline
 spam$_9$    &  41.9999 \\
 \emph<i>eggs</i> & 451      \\
\hline
\end<i>tabular</i>
```

"latex_booktabs" produces a tabular environment of LaTeX document markup
using the booktabs.sty package:

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="latex_booktabs"))
\begin<i>tabular}{lr</i>
\toprule
 spam &  41.9999 \\
 eggs & 451      \\
\bottomrule
\end<i>tabular</i>
```

"latex_longtable" produces a tabular environment that can stretch along
multiple pages, using the longtable package for LaTeX.

```
>>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="latex_longtable"))
\begin<i>longtable}{lr</i>
\hline
 spam &  41.9999 \\
 eggs & 451      \\
\hline
\end<i>longtable</i>
```


Number parsing
--------------
By default, anything which can be parsed as a number is a number.
This ensures numbers represented as strings are aligned properly.
This can lead to weird results for particular strings such as
specific git SHAs e.g. "42992e1" will be parsed into the number
429920 and aligned as such.

To completely disable number parsing (and alignment), use
<b>disable_numparse=True</b>. For more fine grained control, a list column
indices is used to disable number parsing only on those columns
e.g. <b>disable_numparse=[0, 2]</b> would disable number parsing only on the
first and third columns.