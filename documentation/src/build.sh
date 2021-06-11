# create documentation

# use readme as index
cp ../../README.md docs/index.md

# python packages install
pip install -U -r requirements

# build documentation
cd src
mkdocs build