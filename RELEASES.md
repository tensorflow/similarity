# Tensorflow Similarity releases notes

## 0.12.x - Q/A release

This release focuses on on code quality, API consistency, testing, and
documentation to get TF.similarity ready for production.

## Major improvements

- Refactored Class model to allows easy sub-classing and pave
the way to support a wider range of model type that requires different
`train_step()`.

- Improved typing coverage

## Breaking changes

- `model.SimilarityModel` moved to `models/` and made base class to pave the
way to subclass for specific type of similiarty model that requires to override
things like `train_step()`. Import is now: `from tfsim.models import SimilarityModel`


## 0.11.x - Support for TF dataset

This release focuses on adding TF dataset support.

## New features

- Added Efficient Sampling from TF.data.TFRecordDataset now supported via the
TFDatasetSampler.
- masked_minimum and masked_maximum now return the arg_min and arg_max for
each example in the batch.

## Major improvements

- Improved the API documentation

## 0.10.x - Q/A release

This release focuses on code quality, API consistency, and documentation
to get TF.similarity ready for production.

## New features

- Package documentation is available
- Manhattan distance is available.
- nmslib matcher now supports l2 and l1 distances.

## Major improvements

- Simplified the MemorySampler prototype and made them more consistent.
- Distance refactored to be object
- Refactored tensor types
- All major classes are properly documented
- All major classes are typed.

## Major fixes

- Mean metric now properly return fractional ranks.
- Evaluation Callback properly work on multioutput.
- Memory samplers now properly work on arbitrary tensor type.

## 0.9.x - Muli-output support

This release focuses on refactoring the code to support cleanly multi-output
models.

### New features

- `compile()` now support `embedding_output` to indicate which model output to
use for indexing and lookups.

### Major improvements

- Better `hello_world` notebook - improved explaination and simplified.


## 0.8.x - Evaluation and vizualization

This release is a major refactoring the evaluation subsystem to make
it more flexible, better tested and having additional functionality that makes
model evaluation easier.

### New features

- Evaluate()
- Calibrate()

- `pairwise_euclidean()` is available.

- Added an Evaluation callback to easily track metric evolution during
training. Currently the callback can't report metrics in History
due to a TF limitation. Working on finding a solution. Metrics can be tracked
via `TensorBoard`.

- Indexer now supports the `to_pandas()` method to make data analysis
and visualization easier.

### Major improvements

- Full rewrite of the evaluator() subsystem that allows configurable metrics
for evaluation and calibration.

- Metrics are now objects that have access to a clean intermediate data
representation tha makes it easy

- Better types for tensors that distinguish float and int based tensors and
more part of the code is typed.

## 0.7 - Data Samplers

This release focus on improving and extending data samplers.

### New features

- `SingleShotMemorySampler()` is available

- Data samplers now support:
  - The use of a data augmenter
  - Having a warmup period before augmentation

### Major improvements

- `single_lookup()` now returns the rank of each match.
- Memory Samplers now scale to millions of examples without slowdown.
- Rewrote the sampler interface to allows better customization and subclassing.

### Major bug fixes

- Fixed the issue where model couldn't be reloaded due to a bug in load_model()

## 0.6.x - Distance Metrics

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
