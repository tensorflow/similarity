# Overview

Embedding Matchers are used to find the closest embeddings to a query embebbeding. Under the hood they perform a [ANN (Approximate Nearst Neigboors)](https://en.wikipedia.org/wiki/Nearest_neighbor_search) search on the indexed dataset of known embeddings to do so. Different ANN librairies have [different performance profiles](https://github.com/erikbern/ann-benchmarks).

Tensorflow Similarity by default use [NMSLIB](https://github.com/nmslib/nmslib) which has a strong performance profile and is very portable.

Adding another backend is fairly straightforward - create subclass the abstract class Matcher() that implements the needed method and pass it in the `compile()` method of your [SimilarityModel]. Adding it as a built-in choice for the user involves modifiying the [Indexer] and sending a PR. In general, unless the backend if of general use, its better to not include it as a general option.

## Modules

[NMSLibMatcher](nmslib.md): Default Matcher implemented on top of the ANN library [NMSLIB](https://github.com/nmslib/nmslib).

[Matcher](matcher.md): Abstract Matcher class.
