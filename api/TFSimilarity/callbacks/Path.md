# TFSimilarity.callbacks.Path






PurePath subclass that can make system calls.


```python
TFSimilarity.callbacks.Path(
    *args, **kwargs
)
```



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
<b>anchor</b>
</td>
<td>
The concatenation of the drive and root, or ''.
</td>
</tr><tr>
<td>
<b>drive</b>
</td>
<td>
The drive prefix (letter or UNC path), if any.
</td>
</tr><tr>
<td>
<b>name</b>
</td>
<td>
The final path component, if any.
</td>
</tr><tr>
<td>
<b>parent</b>
</td>
<td>
The logical parent of the path.
</td>
</tr><tr>
<td>
<b>parents</b>
</td>
<td>
A sequence of this path's logical parents.
</td>
</tr><tr>
<td>
<b>parts</b>
</td>
<td>
An object providing sequence-like access to the
components in the filesystem path.
</td>
</tr><tr>
<td>
<b>root</b>
</td>
<td>
The root of the path, if any.
</td>
</tr><tr>
<td>
<b>stem</b>
</td>
<td>
The final path component, minus its last suffix.
</td>
</tr><tr>
<td>
<b>suffix</b>
</td>
<td>
The final component's last suffix, if any.

This includes the leading period. For example: '.txt'
</td>
</tr><tr>
<td>
<b>suffixes</b>
</td>
<td>
A list of the final component's suffixes, if any.

These include the leading periods. For example: ['.tar', '.gz']
</td>
</tr>
</table>



## Methods

<h3 id="absolute">absolute</h3>

```python
absolute()
```


Return an absolute version of this path.  This function works
even if the path doesn't point to anything.

No normalization is done, i.e. all '.' and '..' will be kept along.
Use resolve() to get the canonical path to a file.

<h3 id="as_posix">as_posix</h3>

```python
as_posix()
```


Return the string representation of the path with forward (/)
slashes.

<h3 id="as_uri">as_uri</h3>

```python
as_uri()
```


Return the path as a 'file' URI.


<h3 id="chmod">chmod</h3>

```python
chmod(
    mode
)
```


Change the permissions of the path, like os.chmod().


<h3 id="cwd">cwd</h3>

``<b>python
@classmethod</b>``

```python
cwd()
```


Return a new path pointing to the current working directory
(as returned by os.getcwd()).

<h3 id="exists">exists</h3>

```python
exists()
```


Whether this path exists.


<h3 id="expanduser">expanduser</h3>

```python
expanduser()
```


Return a new path with expanded ~ and ~user constructs
(as returned by os.path.expanduser)

<h3 id="glob">glob</h3>

```python
glob(
    pattern
)
```


Iterate over this subtree and yield all existing files (of any
kind, including directories) matching the given relative pattern.

<h3 id="group">group</h3>

```python
group()
```


Return the group name of the file gid.


<h3 id="home">home</h3>

``<b>python
@classmethod</b>``

```python
home()
```


Return a new path pointing to the user's home directory (as
returned by os.path.expanduser('~')).

<h3 id="is_absolute">is_absolute</h3>

```python
is_absolute()
```


True if the path is absolute (has both a root and, if applicable,
a drive).

<h3 id="is_block_device">is_block_device</h3>

```python
is_block_device()
```


Whether this path is a block device.


<h3 id="is_char_device">is_char_device</h3>

```python
is_char_device()
```


Whether this path is a character device.


<h3 id="is_dir">is_dir</h3>

```python
is_dir()
```


Whether this path is a directory.


<h3 id="is_fifo">is_fifo</h3>

```python
is_fifo()
```


Whether this path is a FIFO.


<h3 id="is_file">is_file</h3>

```python
is_file()
```


Whether this path is a regular file (also True for symlinks pointing
to regular files).

<h3 id="is_mount">is_mount</h3>

```python
is_mount()
```


Check if this path is a POSIX mount point


<h3 id="is_reserved">is_reserved</h3>

```python
is_reserved()
```


Return True if the path contains one of the special names reserved
by the system, if any.

<h3 id="is_socket">is_socket</h3>

```python
is_socket()
```


Whether this path is a socket.


<h3 id="is_symlink">is_symlink</h3>

```python
is_symlink()
```


Whether this path is a symbolic link.


<h3 id="iterdir">iterdir</h3>

```python
iterdir()
```


Iterate over the files in this directory.  Does not yield any
result for the special paths '.' and '..'.

<h3 id="joinpath">joinpath</h3>

```python
joinpath(
    *args
)
```


Combine this path with one or several arguments, and return a
new path representing either a subpath (if all arguments are relative
paths) or a totally different path (if one of the arguments is
anchored).

<h3 id="lchmod">lchmod</h3>

```python
lchmod(
    mode
)
```


Like chmod(), except if the path points to a symlink, the symlink's
permissions are changed, rather than its target's.

<h3 id="lstat">lstat</h3>

```python
lstat()
```


Like stat(), except if the path points to a symlink, the symlink's
status information is returned, rather than its target's.

<h3 id="match">match</h3>

```python
match(
    path_pattern
)
```


Return True if this path matches the given pattern.


<h3 id="mkdir">mkdir</h3>

```python
mkdir(
    mode=511, parents=False, exist_ok=False
)
```


Create a new directory at this given path.


<h3 id="open">open</h3>

```python
open(
    mode=r, buffering=-1, encoding=None, errors=None, newline=None
)
```


Open the file pointed by this path and return a file object, as
the built-in open() function does.

<h3 id="owner">owner</h3>

```python
owner()
```


Return the login name of the file owner.


<h3 id="read_bytes">read_bytes</h3>

```python
read_bytes()
```


Open the file in bytes mode, read it, and close the file.


<h3 id="read_text">read_text</h3>

```python
read_text(
    encoding=None, errors=None
)
```


Open the file in text mode, read it, and close the file.


<h3 id="relative_to">relative_to</h3>

```python
relative_to(
    *other
)
```


Return the relative path to another path identified by the passed
arguments.  If the operation is not possible (because this is not
a subpath of the other path), raise ValueError.

<h3 id="rename">rename</h3>

```python
rename(
    target
)
```


Rename this path to the given path.


<h3 id="replace">replace</h3>

```python
replace(
    target
)
```


Rename this path to the given path, clobbering the existing
destination if it exists.

<h3 id="resolve">resolve</h3>

```python
resolve(
    strict=False
)
```


Make the path absolute, resolving all symlinks on the way and also
normalizing it (for example turning slashes into backslashes under
Windows).

<h3 id="rglob">rglob</h3>

```python
rglob(
    pattern
)
```


Recursively yield all existing files (of any kind, including
directories) matching the given relative pattern, anywhere in
this subtree.

<h3 id="rmdir">rmdir</h3>

```python
rmdir()
```


Remove this directory.  The directory must be empty.


<h3 id="samefile">samefile</h3>

```python
samefile(
    other_path
)
```


Return whether other_path is the same or not as this file
(as returned by os.path.samefile()).

<h3 id="stat">stat</h3>

```python
stat()
```


Return the result of the stat() system call on this path, like
os.stat() does.

<h3 id="symlink_to">symlink_to</h3>

```python
symlink_to(
    target, target_is_directory=False
)
```


Make this path a symlink pointing to the given path.
Note the order of arguments (self, target) is the reverse of os.symlink's.

<h3 id="touch">touch</h3>

```python
touch(
    mode=438, exist_ok=True
)
```


Create this file with the given access mode, if it doesn't exist.


<h3 id="unlink">unlink</h3>

```python
unlink()
```


Remove this file or link.
If the path is a directory, use rmdir() instead.

<h3 id="with_name">with_name</h3>

```python
with_name(
    name
)
```


Return a new path with the file name changed.


<h3 id="with_suffix">with_suffix</h3>

```python
with_suffix(
    suffix
)
```


Return a new path with the file suffix changed.  If the path
has no suffix, add given suffix.  If the given suffix is an empty
string, remove the suffix from the path.

<h3 id="write_bytes">write_bytes</h3>

```python
write_bytes(
    data
)
```


Open the file in bytes mode, write to it, and close the file.


<h3 id="write_text">write_text</h3>

```python
write_text(
    data, encoding=None, errors=None
)
```


Open the file in text mode, write to it, and close the file.


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
    t, v, tb
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


<h3 id="__le__">__le__</h3>

```python
__le__(
    other
)
```


Return self<=value.


<h3 id="__lt__">__lt__</h3>

```python
__lt__(
    other
)
```


Return self<value.


<h3 id="__rtruediv__">__rtruediv__</h3>

```python
__rtruediv__(
    key
)
```





<h3 id="__truediv__">__truediv__</h3>

```python
__truediv__(
    key
)
```







