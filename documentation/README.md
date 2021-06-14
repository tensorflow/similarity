# TF Similarity documentation

The user-facing documentation is available FIXME - this is what you are looking for if you are a package user. This directory contains the source used to generate the documentation and are not easily readable as is.


# Generating the documentation


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


## generating local documentation

To generate the documentation in `site/` run the  `build.sh` script.

## generating public documentation

A github action do this for the project on action merged.