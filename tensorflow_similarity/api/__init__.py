# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

"""

![TensorFlow Similarity Overview](../assets/images/tfsim_overview.png)

TensorFlow Similiarity, as visible in the diagram above, offers the following
components to help research, train, evaluate and serve metric models:

- **`SimilarityModel()`**: This class subclasses the `tf.keras.model` class and
  extends it with additional properties that are useful for metric learning. For
  example it adds the methods: 1. `index()`: Enables indexing of the embedding
  2. `lookup()`: Takes samples, calls predict(), and searches for neighbors
  within the index.  3. `calibrate()`: Calibrates the model's index search
  thresholds using a calibration metric and a test dataset.

- **`MetricLoss()`**:  This virtual class, that extends the `tf.keras.Loss`
  class, is the base class from which Metric losses are derived. This
  sub-classing ensures proper error checking; that is, it ensures the user is
  using a loss metric to train the models, performs better static analysis, and
  enforces additional constraints such as having a distance function that is
  supported by the index. Additionally, Metric losses make use of the fully
  tested and highly optimized pairwise distances functions provided by
  TensorFlow Similarity that are available under the `Distances.*` classes.

- **`Samplers()`**: Samplers are meant to ensure that each batch has at least n
  (with n >=2) examples of each class, as losses such as TripletLoss can’t work
  properly if this condition is not met. TensorFlow Similarity offers an
  in-memory sampler for small dataset and a `tf.data.TFRecordDataset` for large
  scales one.

- **`Indexer()`**: The Indexer and its sub-components are meant to index known
  embeddings alongside their metadata. The embedding metadata is stored within
  `Table()`, while the `Matcher()` is used to perform [fast approximate neighbor
          searches](https://en.wikipedia.org/wiki/Nearest_neighbor_search) that
  are meant to quickly retrieve the indexed elements that are the closest to the
  embeddings supplied in the `lookup()` and `single_lookup()` function.

The default `Index()` sub-compoments run in-memory and are optimized to be used
in interactive settings such as Jupyter notebooks, Colab, and metric computation
during training (e.g using the `EvalCallback()` provided). Index are serialized
as part of `model.save()` so you can reload them via `model.index_load()` for
serving purpose or further training / evaluation.

The default implementation can scale up to medium deployment (1M-10M+ points)
easily, provided the computers have enough memory. For very large scale
deployments you will need to sublcass the compoments to match your own
architetctue. See FIXME colab to see how to deploy TensorFlow Similarity in
production.
"""

# The file is used to list the public API for document generation purpose.
from tensorflow_similarity import architectures  # noqa
from tensorflow_similarity import augmenters  # noqa
from tensorflow_similarity import callbacks  # noqa
from tensorflow_similarity import classification_metrics  # noqa
from tensorflow_similarity import distances  # noqa
from tensorflow_similarity import evaluators  # noqa
from tensorflow_similarity import indexer  # noqa
from tensorflow_similarity import layers  # noqa
from tensorflow_similarity import losses  # noqa
from tensorflow_similarity import matchers  # noqa
from tensorflow_similarity import models  # noqa
from tensorflow_similarity import retrieval_metrics  # noqa
from tensorflow_similarity import samplers  # noqa
from tensorflow_similarity import search  # noqa
from tensorflow_similarity import stores  # noqa
from tensorflow_similarity import training_metrics  # noqa
from tensorflow_similarity import utils  # noqa
from tensorflow_similarity import visualization  # noqa

# from tensorflow_similarity import algebra  # not public
# from tensorflow_similarity import types  # not public
