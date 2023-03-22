# Changelog

All notable changes to this project will be documented in this file.

## [0.17.x] - Additional Loss Functions, Refactors, and Bug Fixes

Thanks to:
- @yonigottesman 
- @abhisharsinha 
- @aminhp
- @dewball345

### Added

- Contrastive model now provides a create_contrastive_model() function that provides default projector and predictor MLP models.
- Full support for all nmslib parameters in the nmslib search class. Models now correctly serialize and load custom nmslib configurations.
- support for MultiShotFileSample to enable passing in a data loader function to read examples from disk.
- Precision@K to the retrieval metrics. See section 3.2 of https://arxiv.org/pdf/2003.08505.pdf
- support for retrieval metrics in the metrics callback.
- support for clipping the retrieval metric at R per class. This masks out any values past the R expected neighbors.
- Added VicReg Loss to contrastive losses.
- Add support for passing the same examples for both the query and indexed set when calling retrieval_metrics. Added a new param to each retrieval_metric that enables dropping the nearest neighbor. This is useful if the nearest neighbor exists in the indexed examples.
- New Notebooks for transfer learning new Clip embeddings and Examples for using Retrieval metrics.

### Changed

- Contrastive model now saves and loads using the standard keras methods.
- Removed references to multiple output heads in Contrastive model.
- Update all typing to use standard collections (see pep 585).
- Start refactoring tests to improve coverage and use TF TestCase
- Architectures now use syncbatchnorm instead of batchnorm to support distributed training.
- Refactored image augmentation utilities.

### Fixed

- Contrastive model loading verified to work correctly with TF 2.8 to 2.11
- SimClr now works correctly with distributed training.
- Add better support for mixed precision and float16 dtype policies, and cast from local tensor dtypes when possible.
- Speed up tests and add support for testing python 3.8 and 3.10 cross TF 2.7 and 2.11.
- Warmup Cosine LR Schedule now correctly reaches the max LR level and the cosine decay begins after reaching the max value.
- Other small fixes and improvements


## [0.16.x] - Additional Loss Functions, Refactors, and Bug Fixes 

### Added

- Cross-batch memory (XBM). Thanks @chjort 
- VicReg Loss - Improvement of Barlow Twins. Thanks @dewball345 
- Add augmenter function for Barlow Twins. Thanks @dewball345

### Changed

- Simplified MetricEmbedding layer. Function tracing and serialization are better supported now.
- Refactor image augmentation modules into separate utils modules to help make them more composable. Thanks @dewball345
- GeneralizedMeanPooling layers default value for P is now 3.0. This better aligns with the value in the paper.
- EvalCallback now supports split validation callback. Thanks @abhisharsinha 
- Distance and losses refactor. Refactor distances call signature to now accept query and key inputs instead of a single set of embeddings or labels. Thanks @chjort


### Fixed

- Fix TFRecordDatasetSampler to ensure the correct number of examples per class per batch. Deterministic is now set to True and we have updated the docstring to call out the requirements for the tf record files.
- Removed unneeded tf.function and registar_keras_serializable decorators.
- Refactored the model index attribute to raise a more informative AttributeError if the index does not exist.
- Freeze all BatchNormalization layers in architectures when loading weights.
- Fix bug in losses.utils.LogSumExp(). tf.math.log(1 + x) should be tf.math.log(tf.math.exp(-my_max) + x). This is needed to properly account for removing the row wise max before computing the logsumexp. 
- Fix multisim loss offsets. The tfsim version of multisim uses distances instead of the inner product. However, multisim requires that we "center" the pairwise distances around 0. Here we add a new center param, which we set to 1.0 for cosine distance. Additionally, we also flip the lambda (lmda) param to add the threshold to the values instead of subtracting it. These changes will help improve the pos and neg weighting in the log1psumexp.
- Fix nmslib save and load. nmslib requires a string path and will only read and write to local files. In order to support writing to a remote file path, we first write to a local tmp dir and then write that to the user provided path using tf.io.gfile.GFile.
- Fix serialization of Simclr params in get_config()
- Other fixes and improvements...

## [0.15.x] - Initial support of self supervised algorithms

### Added

- Refactored Augmenters to be a class
- Added SimCLRAugmenter
- Added `SiamSiamLoss()`
- Added SimCLR Loss
- Added `ContrastiveModel()` for self-supervised training
- Added `encoder_dev()` metrics as suggested in SiamSiam to detect collapsing
- Added `visualize_views()` to see view side by side
- Added self-supervised hello world narrated notebook
- Refactored `Augmenter()` as class.

### Changed

- Remove augmentation argument from architectures as the augmentation arg could lead to issues when saving the model or training on TPU. 

## [0.14.x] - Post Launch bugfix release

### Added

- Updates to the TFRecordDataset sampler.

### Changed

- TFRecordDatasetSampler:
  1. Remove file_parallelism param.
  2. Add async_cycle param that now defaults to setting the number of threads for the input cycle shards to 1. 
     This is more stable and generally faster than setting multiple threads, however, setting async_cycle=True 
     will now set the num_parallel_calls eqal to the cycle_length.

### Fixed

- Many bugs fixed based on the community feedback. Thanks!

## [0.13.x] - Launch release

Finalizing all the features planned for initial release, including visulization tools, intergation with TensorFlow dataset catalog and additional losses and distances

Refactored the Index API to prepare for major overhaul while keeping it transparent.

Refactored the evaluation metrics to support both classification and retrieval metrics.


### Added

- Interactive notebook projector vizualization added. Works on notebooks and colab. Allows to inspect to project embeddings and inspect how well the data is separated.
- New distances:
  1. Inner Product distance
  2. Squared Euclidean
- New losses:
  1. [PN loss](https://doi.org/10.1038/s41598-019-45301-0)
  2. [Circle loss](https://arxiv.org/pdf/2002.10857.pdf)
- TensorFlow Catalog integration via `TFDatasetMultiShotMemorySampler()`
- New retrieval evaluation metrics:
  1. recall@k
  2. precision@k
  3. map@k
  4. binary ndcg@k

### Changed

- The `__init__()` of the matcher has been refactored to be implementation agnostic. Should not have any impact on user facing API but makes it incompatible with previous internal components.
- The `__init__()` of the Indexer API has been refactored. Will break code that directly calls it.
- select_samples() arguments were improved to be clearer, breaks previous code using them but should requires only to rename a variable or two to work.
- The calibration and evaluation methods now call the classification metrics and accept a Matcher object that maps a set of lookup results to a single label and distance.

### Fixed

- Distance aliasing is now delegated to the Distance implementation which makes it more modular and removed many bugs due to naming duplication.
- Allows to configure the number of shards used during an dataset interleave cycle via `shard_per_cycle` in TFRecordDataset.
- Better Index serialization meta_data that now include the size of the index and if it was compressed.
- Evaluation metrics are now vectorized using Tensorflow.

## [0.12.x] - Q/A release

This release focuses on on code quality, API consistency, testing, and
documentation to get TF.similarity ready for production.

### Added

- Advanced colab with use of callbacks and TFRecordDataset sampler added. (v0.12.6)
- Callback to track progress on known vs unknown classes added. (v0.12.6)

### Fixed

- Multi-thread `lookup()` working with linear scale speedup in number of core used. (v0.12.8)
- Refactored the `index()` to support storing arbitrary Tensor as data in the index instead of being restricted to model input. (v0.12.5)
- Refactored Class model to allows easy sub-classing and pave the way to support a wider range of model type that requires different `train_step()`. (v0.12.1)
- Improved typing coverage (v0.12.2)
- Lookup results are now arrays of Lookup() type with attributes. (v0.12.0)

### Changed

- `model.SimilarityModel` moved to `models/` and made base class to pave the way to subclass for specific type of similiarty model that requires to override things like `train_step()`. Import is now: `from tfsim.models import SimilarityModel` (v0.12.0)

## [0.11.x] - Support for TF dataset

This release focuses on adding TF dataset support.

### Added

- Added Efficient Sampling from TF.data.TFRecordDataset now supported via the TFDatasetSampler.
- masked_minimum and masked_maximum now return the arg_min and arg_max for each example in the batch.

### Fixed

- Improved the API documentation

## [0.10.x] - Q/A release

This release focuses on code quality, API consistency, and documentation to get TF.similarity ready for production.

### Added

- Package documentation is available
- Manhattan distance is available.
- nmslib matcher now supports l2 and l1 distances.

### Fixed

- Simplified the MemorySampler prototype and made them more consistent.
- Distance refactored to be object
- Refactored tensor types
- All major classes are properly documented
- All major classes are typed.
- Mean metric now properly return fractional ranks.
- Evaluation Callback properly work on multioutput.
- Memory samplers now properly work on arbitrary tensor type.

## [0.9.x] - Muli-output support

This release focuses on refactoring the code to support cleanly multi-output models.

### Added

- `compile()` now support `embedding_output` to indicate which model output to use for indexing and lookups.

### Fixed

- Better `hello_world` notebook - improved explaination and simplified.

## [0.8.x] - Evaluation and vizualization

This release is a major refactoring the evaluation subsystem to make it more flexible, better tested and having additional functionality that makes model evaluation easier.

### Added

- Evaluate()
- Calibrate()
- `pairwise_euclidean()` is available.
- Added an Evaluation callback to easily track metric evolution during training. Currently the callback can't report metrics in History due to a TF limitation. Working on finding a solution. Metrics can be tracked via `TensorBoard`.
- Indexer now supports the `to_pandas()` method to make data analysis and visualization easier.

### Fixed

- Better types for tensors that distinguish float and int based tensors and more part of the code is typed.

### Changed

- Full rewrite of the evaluator() subsystem that allows configurable metrics for evaluation and calibration.
- Metrics are now objects that have access to a clean intermediate data representation tha makes it easy

## [0.7] - Data Samplers

This release focus on improving and extending data samplers.

### Added

- `SingleShotMemorySampler()` is available
- Data samplers now support:
  - The use of a data augmenter
  - Having a warmup period before augmentation
- `single_lookup()` now returns the rank of each match.
- Memory Samplers now scale to millions of examples without slowdown.

### Fixes

- Fixed the issue where model couldn't be reloaded due to a bug in load_model()

### Changed

- Rewrote the sampler interface to allows better customization and subclassing.

## [0.6.x] - Distance Metrics

This release add specialized distance metrics and fixes bugs

### Added

- Distance metrics are now avialable under `tensorflow_similarity.distance_metrics` including:
  - `dist_gap`: distance between the maximum positive distance and the min negative distance.
  - `pos_max`: positive maximum distance
  - `neg_min`: negative minimal distance.
- More generally minimal, average and maximal distance for positive and negative anchors are supported. You can either uses the short-hand alias (e.g `neg_min`, `pos_avg`....), or use the underlying `DistanceMetric()` class that allows to configure all aspect of the metric.

## [0.5] - Save & load

This release focus on a major refactor of the internal API to ensure clean decoupling and ensure the package offers streamlined API that makes it easy to extend the package to additional backend.  

### Added

- Ability to save and reload the model from disk via:
  - save: `model.save()`, `index_save()`
  - load: `model.load()`, `index_load()`
- Integration tests and many unit tests added to ensure that core packages features are robusts and wont' suffer major regression as revision are released. C/I added as well for automatic deployement.

### Fixed

- `Matcher` and `Table` API streamlined to avoid unnecessary data manipulation that results in increased performance.

### Changed

- Model API renamed to be more consistent with Keras naming convention to have model verbs. For example: `index_reset()` was renamed `reset_index()`.
- The indexer was refactored to decouple the matcher, index table and matching logic to be in different components that can be transparently replaced with alternatives that implements the abstract class methods.
- `Evaluator` API introduced with major refactoring to the `Model` and `Indexer` class to ensure all the embedding evaluation code is now self contained and have clean interface.

## [0.4.1]

This release focuses on making the package installable via pip and providing a well documented hello world colab.

Initial fully functional alpha release based on the new core engine that leverages TF 2.x improvements over TF 1.x. This release focuses on showcasing the new core API to collect feedback and comments.
