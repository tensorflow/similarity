# Module: TFSimilarity.search





Efficiently find nearest indexed embeddings


The search is used to find the closest indexed example embeddings
to a query example embebbeding.
To do so it performs a sub-linear time
- [ANN (Approximate Nearst Neigboors)](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search on the indexed set of embedding examples.

Different ANN librairies have [different performance profiles](https://github.com/erikbern/ann-benchmarks).
Tensorflow Similarity by default use [NMSLIB](https://github.com/nmslib/nmslib)
which has a strong performance profile and is very portable.

Adding another backend is fairly straightforward: subclass the
abstract class `Search()` and implement the abstract methods. Then to use it
pass it to the `compile()` method of your [SimilarityModel].

Adding your search backend as a built-in choice invlolves
modifiying the [Indexer](../indexer.md) and sending a PR. In general, unless
the backend is of general use, its better to not include it as
a built-in option as it must be supported moving forward.

## Classes

- [`class NMSLibSearch`](../TFSimilarity/indexer/NMSLibSearch.md): Efficiently find nearest embeddings by indexing known embeddings and make

- [`class Search`](../TFSimilarity/indexer/Search.md): Helper class that provides a standard way to create an ABC using

