# Module: TFSimilarity.matchers





Matchers define the classification matching strategy when using

similarity models to match query examples to the classes of indexed
examples.

## Classes

- [`class ClassificationMatch`](../TFSimilarity/callbacks/ClassificationMatch.md): Abstract base class for defining the classification matching strategy.

- [`class MatchMajorityVote`](../TFSimilarity/matchers/MatchMajorityVote.md): Match metrics for the most common label in a result set.

- [`class MatchNearest`](../TFSimilarity/matchers/MatchNearest.md): Match metrics for labels at k=1.

## Functions

- [`make_classification_matcher(...)`](../TFSimilarity/indexer/make_classification_matcher.md): Convert classification matcher from str name to object if needed.

