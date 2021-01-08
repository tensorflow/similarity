#!/bin/bash


SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd $SOURCE_DIR
rm -rf dist/
python update_version.py
python setup.py sdist && twine upload --verbose -r google dist/*
popd


