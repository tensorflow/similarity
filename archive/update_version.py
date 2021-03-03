import re

VERSION_RE = re.compile(r'(\d+) \. (\d+) (\. (\d+))? ([ab]\d+)?',
                        re.VERBOSE | re.ASCII)
FILE_PATH = "tensorflow_similarity/__init__.py"

with open(FILE_PATH, "rt") as i:
    text = ''
    for line in i.readlines():
        if line.startswith('__version__'):
            match = VERSION_RE.search(line)
            (major, minor, patch, prerelease) = match.group(1, 2, 4, 5)
            patch = str(int(patch) + 1) if patch else 0
            # Updating the patch number drops the prerelease value.
            version = f'{major}.{minor}.{patch}'
            line = f"__version__ = '{version}'\n"
        text += line

with open(FILE_PATH, "wt") as o:
    o.write(text)
