# Module: TFSimilarity.stores





Key Values Stores store the data associated with the embeddings indexed by

the `Indexer()`.

Each key of the store represent a **record** that contains information
about a given embedding.

The main use-case for the store is to retrieve the records associated
with the ids returned by a nearest neigboor search performed with the
- [`Search()`](../search/).

Additionally one might want to inspect the content of the index which is why
`Store()` class may implement an export to
a [Pandas Dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
via the `to_pandas()` method.

## Classes

- [`class MemoryStore`](../TFSimilarity/indexer/MemoryStore.md): Efficient in-memory dataset store

- [`class Store`](../TFSimilarity/indexer/Store.md): Helper class that provides a standard way to create an ABC using

