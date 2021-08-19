# TFSimilarity.callbacks.Path
<!-- Insert buttons and diff -->
<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
</table>

PurePath subclass that can make system calls.
<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`TFSimilarity.indexer.Path`</p>
</p>
</section>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>TFSimilarity.callbacks.Path(
    *args, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->
Path represents a filesystem path but unlike PurePath, also offers
methods to do system calls on path objects. Depending on your system,
instantiating a Path will return either a PosixPath or a WindowsPath
object. You can also instantiate a PosixPath or WindowsPath directly,
but cannot instantiate a WindowsPath on a POSIX system or vice versa.
<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`anchor`
</td>
<td>
The concatenation of the drive and root, or ''.
</td>
</tr><tr>
<td>
`drive`
</td>
<td>
The drive prefix (letter or UNC path), if any.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The final path component, if any.
</td>
</tr><tr>
<td>
`parent`
</td>
<td>
The logical parent of the path.
</td>
</tr><tr>
<td>
`parents`
</td>
<td>
A sequence of this path's logical parents.
</td>
</tr><tr>
<td>
`parts`
</td>
<td>
An object providing sequence-like access to the
components in the filesystem path.
</td>
</tr><tr>
<td>
`root`
</td>
<td>
The root of the path, if any.
</td>
</tr><tr>
<td>
`stem`
</td>
<td>
The final path component, minus its last suffix.
</td>
</tr><tr>
<td>
`suffix`
</td>
<td>
The final component's last suffix, if any.
This includes the leading period. For example: '.txt'
</td>
</tr><tr>
<td>
`suffixes`
</td>
<td>
A list of the final component's suffixes, if any.
These include the leading periods. For example: ['.tar', '.gz']
</td>
</tr>
</table>


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>
<tr>
<td>
`anchor`
</td>
<td>
The concatenation of the drive and root, or ''.
</td>
</tr><tr>
<td>
`drive`
</td>
<td>
The drive prefix (letter or UNC path), if any.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The final path component, if any.
</td>
</tr><tr>
<td>
`parent`
</td>
<td>
The logical parent of the path.
</td>
</tr><tr>
<td>
`parents`
</td>
<td>
A sequence of this path's logical parents.
</td>
</tr><tr>
<td>
`parts`
</td>
<td>
An object providing sequence-like access to the
components in the filesystem path.
</td>
</tr><tr>
<td>
`root`
</td>
<td>
The root of the path, if any.
</td>
</tr><tr>
<td>
`stem`
</td>
<td>
The final path component, minus its last suffix.
</td>
</tr><tr>
<td>
`suffix`
</td>
<td>
The final component's last suffix, if any.
This includes the leading period. For example: '.txt'
</td>
</tr><tr>
<td>
`suffixes`
</td>
<td>
A list of the final component's suffixes, if any.
These include the leading periods. For example: ['.tar', '.gz']
</td>
</tr>
</table>

## Methods
<h3 id="absolute"><code>absolute</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>absolute()
</code></pre>
Return an absolute version of this path.  This function works
even if the path doesn't point to anything.
No normalization is done, i.e. all '.' and '..' will be kept along.
Use resolve() to get the canonical path to a file.
<h3 id="as_posix"><code>as_posix</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_posix()
</code></pre>
Return the string representation of the path with forward (/)
slashes.
<h3 id="as_uri"><code>as_uri</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_uri()
</code></pre>
Return the path as a 'file' URI.

<h3 id="chmod"><code>chmod</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>chmod(
    mode
)
</code></pre>
Change the permissions of the path, like os.chmod().

<h3 id="cwd"><code>cwd</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>cwd()
</code></pre>
Return a new path pointing to the current working directory
(as returned by os.getcwd()).
<h3 id="exists"><code>exists</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>exists()
</code></pre>
Whether this path exists.

<h3 id="expanduser"><code>expanduser</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expanduser()
</code></pre>
Return a new path with expanded ~ and ~user constructs
(as returned by os.path.expanduser)
<h3 id="glob"><code>glob</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>glob(
    pattern
)
</code></pre>
Iterate over this subtree and yield all existing files (of any
kind, including directories) matching the given relative pattern.
<h3 id="group"><code>group</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>group()
</code></pre>
Return the group name of the file gid.

<h3 id="home"><code>home</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>home()
</code></pre>
Return a new path pointing to the user's home directory (as
returned by os.path.expanduser('~')).
<h3 id="is_absolute"><code>is_absolute</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_absolute()
</code></pre>
True if the path is absolute (has both a root and, if applicable,
a drive).
<h3 id="is_block_device"><code>is_block_device</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_block_device()
</code></pre>
Whether this path is a block device.

<h3 id="is_char_device"><code>is_char_device</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_char_device()
</code></pre>
Whether this path is a character device.

<h3 id="is_dir"><code>is_dir</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_dir()
</code></pre>
Whether this path is a directory.

<h3 id="is_fifo"><code>is_fifo</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_fifo()
</code></pre>
Whether this path is a FIFO.

<h3 id="is_file"><code>is_file</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_file()
</code></pre>
Whether this path is a regular file (also True for symlinks pointing
to regular files).
<h3 id="is_mount"><code>is_mount</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_mount()
</code></pre>
Check if this path is a POSIX mount point

<h3 id="is_reserved"><code>is_reserved</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_reserved()
</code></pre>
Return True if the path contains one of the special names reserved
by the system, if any.
<h3 id="is_socket"><code>is_socket</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_socket()
</code></pre>
Whether this path is a socket.

<h3 id="is_symlink"><code>is_symlink</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symlink()
</code></pre>
Whether this path is a symbolic link.

<h3 id="iterdir"><code>iterdir</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iterdir()
</code></pre>
Iterate over the files in this directory.  Does not yield any
result for the special paths '.' and '..'.
<h3 id="joinpath"><code>joinpath</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>joinpath(
    *args
)
</code></pre>
Combine this path with one or several arguments, and return a
new path representing either a subpath (if all arguments are relative
paths) or a totally different path (if one of the arguments is
anchored).
<h3 id="lchmod"><code>lchmod</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lchmod(
    mode
)
</code></pre>
Like chmod(), except if the path points to a symlink, the symlink's
permissions are changed, rather than its target's.
<h3 id="link_to"><code>link_to</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>link_to(
    target
)
</code></pre>
Make the target path a hard link pointing to this path.
Note this function does not make this path a hard link to *target*,
despite the implication of the function and argument names. The order
of arguments (target, link) is the reverse of Path.symlink_to, but
matches that of os.link.
<h3 id="lstat"><code>lstat</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>lstat()
</code></pre>
Like stat(), except if the path points to a symlink, the symlink's
status information is returned, rather than its target's.
<h3 id="match"><code>match</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>match(
    path_pattern
)
</code></pre>
Return True if this path matches the given pattern.

<h3 id="mkdir"><code>mkdir</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mkdir(
    mode=511, parents=(False), exist_ok=(False)
)
</code></pre>
Create a new directory at this given path.

<h3 id="open"><code>open</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>open(
    mode=&#x27;r&#x27;, buffering=-1, encoding=None, errors=None, newline=None
)
</code></pre>
Open the file pointed by this path and return a file object, as
the built-in open() function does.
<h3 id="owner"><code>owner</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>owner()
</code></pre>
Return the login name of the file owner.

<h3 id="read_bytes"><code>read_bytes</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read_bytes()
</code></pre>
Open the file in bytes mode, read it, and close the file.

<h3 id="read_text"><code>read_text</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read_text(
    encoding=None, errors=None
)
</code></pre>
Open the file in text mode, read it, and close the file.

<h3 id="relative_to"><code>relative_to</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>relative_to(
    *other
)
</code></pre>
Return the relative path to another path identified by the passed
arguments.  If the operation is not possible (because this is not
a subpath of the other path), raise ValueError.
<h3 id="rename"><code>rename</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>rename(
    target
)
</code></pre>
Rename this path to the target path.
The target path may be absolute or relative. Relative paths are
interpreted relative to the current working directory, *not* the
directory of the Path object.
Returns the new Path instance pointing to the target path.
<h3 id="replace"><code>replace</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace(
    target
)
</code></pre>
Rename this path to the target path, overwriting if that path exists.
The target path may be absolute or relative. Relative paths are
interpreted relative to the current working directory, *not* the
directory of the Path object.
Returns the new Path instance pointing to the target path.
<h3 id="resolve"><code>resolve</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>resolve(
    strict=(False)
)
</code></pre>
Make the path absolute, resolving all symlinks on the way and also
normalizing it (for example turning slashes into backslashes under
Windows).
<h3 id="rglob"><code>rglob</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>rglob(
    pattern
)
</code></pre>
Recursively yield all existing files (of any kind, including
directories) matching the given relative pattern, anywhere in
this subtree.
<h3 id="rmdir"><code>rmdir</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>rmdir()
</code></pre>
Remove this directory.  The directory must be empty.

<h3 id="samefile"><code>samefile</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>samefile(
    other_path
)
</code></pre>
Return whether other_path is the same or not as this file
(as returned by os.path.samefile()).
<h3 id="stat"><code>stat</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stat()
</code></pre>
Return the result of the stat() system call on this path, like
os.stat() does.
<h3 id="symlink_to"><code>symlink_to</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>symlink_to(
    target, target_is_directory=(False)
)
</code></pre>
Make this path a symlink pointing to the target path.
Note the order of arguments (link, target) is the reverse of os.symlink.
<h3 id="touch"><code>touch</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>touch(
    mode=438, exist_ok=(True)
)
</code></pre>
Create this file with the given access mode, if it doesn't exist.

<h3 id="unlink"><code>unlink</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unlink(
    missing_ok=(False)
)
</code></pre>
Remove this file or link.
If the path is a directory, use rmdir() instead.
<h3 id="with_name"><code>with_name</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_name(
    name
)
</code></pre>
Return a new path with the file name changed.

<h3 id="with_suffix"><code>with_suffix</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_suffix(
    suffix
)
</code></pre>
Return a new path with the file suffix changed.  If the path
has no suffix, add given suffix.  If the given suffix is an empty
string, remove the suffix from the path.
<h3 id="write_bytes"><code>write_bytes</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write_bytes(
    data
)
</code></pre>
Open the file in bytes mode, write to it, and close the file.

<h3 id="write_text"><code>write_text</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write_text(
    data, encoding=None, errors=None
)
</code></pre>
Open the file in text mode, write to it, and close the file.

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
    t, v, tb
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

<h3 id="__le__"><code>__le__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other
)
</code></pre>
Return self<=value.

<h3 id="__lt__"><code>__lt__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>
Return self<value.

<h3 id="__rtruediv__"><code>__rtruediv__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rtruediv__(
    key
)
</code></pre>


<h3 id="__truediv__"><code>__truediv__</code></h3>
<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    key
)
</code></pre>



