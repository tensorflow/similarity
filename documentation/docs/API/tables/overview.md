# Overview

Tables are used to store the data associated with the embeddings stored in the model [Index](../indexer.md). Each row of the table represent a **record** that contains information about a given embedding.

The main use-case for the table is to allows to retrieve the records associated
with the ids returned by a nearest neigboor search done with the [Matcher](../matchers/overivew.md) as part of finding related embeddings. See the [Overview](../overview.md)
for more details.

Additionally one might want to inspect the content of the index which is why
each `Table()` Object implement an export to a [Pandas Dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) via
the `to_pandas()` method.

## Modules

[MemoryTable](memory.md): Default Table implemented as an in-memory object.
Allows very fast retrieval that scale well up to a few millions points.

[Table](table.md): Abstract Table based class.
