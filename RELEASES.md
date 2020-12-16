# Tensorflow Similarity releases notes

## 0.7 - Data Samplers

This release focus on improving and extending data samplers.

### New features

- SingleShotMemorySampler is available

### Major improvement

- Rewrote the sampler interface to allows better customization and subclassing.

## 0.6 - Distance Metrics

This release add specialized distance metrics and fixes bugs

### New features

Distance metrics are now avialable under `tensorflow_similarity.distance_metrics`
including:
- `dist_gap`: distance between the maximum positive distance and the min negative distance.
- `pos_max`: positive maximum distance
- `neg_min`: negative minimal distance.

More generally minimal, average and maximal distance for positive and negative anchors are supported.
You can either uses the short-hand alias (e.g `neg_min`, `pos_avg`....), or use
the underlying `DistanceMetric()` class that allows to configure all aspect of the metric.

## 0.5 - Save & load

This release focus on a major refactor of the internal API to ensure clean
decoupling and ensure the package offers streamlined API that
makes it easy to extend the package to additional backend.

### New features

Ability to save and reload the model from disk via:

- save: `model.save()`, `index_save()`
- load: `model.load()`, `index_load()`

### Breaking changes

Model API renamed to be more consistent with Keras naming convention to have model verbs. For example: `index_reset()` was renamed `reset_index()`.

### Major improvements

- The indexer was refactored to decouple the matcher, index table and matching logic to be in different components that can be transparently replaced with alternatives that implements the abstract class methods.

- `Matcher` and `Table` API streamlined to avoid unnecessary data manipulation that results in increased performance.

- `Evaluator` API introduced with major refactoring to the `Model` and `Indexer` class to ensure all the embedding evaluation code is now self contained and have clean interface.

- Integration tests and many unit tests added to ensure that core packages
features are robusts and wont' suffer major regression as revision are
released. C/I added as well for automatic deployement.

## 0.4.1

This release focuses on making the package installable via pip and providing a well documented hello world colab.

Initial fully functional alpha release based on the new core engine that
leverages TF 2.x improvements over TF 1.x. This release focuses on
showcasing the new core API to collect feedback and comments.
