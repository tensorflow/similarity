# TF Similarity documentation

Use (https://www.mkdocs.org/getting-started/)[mkdocs] with (https://squidfunk.github.io/mkdocs-material/getting-started/)[mkdocs material design them]

## Installing dependencies

```
pip install -r requirements.txt
```


## Serving

When developing the docs you can use mkdocs live reload as follow:

```
mkdocs serve
```


# generate the documentation

To generate the documentation in `documentation/site/` you need to run
from the `documentation/src/` directory:

```
mkdocs build
```

A github action do this for the project on action merged.