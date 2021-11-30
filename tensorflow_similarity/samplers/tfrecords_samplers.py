# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Callable, Optional

import tensorflow as tf
from absl import logging


def TFRecordDatasetSampler(
    shard_path: str,
    deserialization_fn: Callable,
    example_per_class: int = 2,
    batch_size: int = 32,
    shards_per_cycle: int = None,
    compression: Optional[str] = None,
    parallelism: int = tf.data.AUTOTUNE,
    async_cycle: bool = False,
    prefetch_size: Optional[int] = None,
    shard_suffix: str = "*.tfrec",
    num_repeat: int = -1,
) -> tf.data.Dataset:
    """Create a [TFRecordDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) based sampler.

    This sampler should be used when using a TFDataset or have a large
    dataset that needs to be stored on file.

    **WARNING**: This samplers assume that classes examples are contigious,
    at least enough that you can get `example_per_class` numbers
    of them consecutively. This requirements is needed to make the
    sampling efficient and makes dataset constuctionn oftentime easier as
    there is no need to worry about shuffling. Somewhat contigious means
    its fine to have the same class in multiples shards as long as the
    examples for the same classes are contigious in that shard.

    Overall the way we make the sampling process is by using the
    [tf.dataset.interleaves](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
    in a non orthodox way: we use its `block_length` to control the
    number of example per class and rely on the parallelize &
    non_deterministic version of the API to do the sampling efficiently
    for us. Relying on pure tf.data ops also ensure good compatibility with
    distribution strategy.


    Args:
        shard_path: Directory where the shards are stored.

        deserialization_fn: Function used to deserialize the tfRecord and
        construct a valid example.

        example_per_class: Number of example per class in each batch.
        Defaults to 2.

        batch_size: How many examples in each batch. The number of class in
        the batch will be `batch_size // example_per_class`.
        Defaults to 32.

        shards_per_cycle: How many shards to use concurrently per cycle.
        Default is None which is all of them. Can cause segv if too many
        shards.

        compression: Which compression was used when creating the dataset.
        `{None, "ZLIB", or "GZIP"}` as specified in [TFRecordDataset documentation](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
        Defaults to None.

        parallelism: How many parallel calls to do. If not set, will let
        TensorFlow decide by using `tf.data.AUTOTUNE` (-1).

        async_cycle: If True, create a threadpool of size `batch_size //
        example_per_class` and fetch inputs from the cycle shards
        asynchronously; however, in practice, the default single thread setting
        is faster. We only recommend setting this to True if it is absolutely
        necessary.

        prefetch_size: How many batch to precache. Defaults to 10.

        shard_suffix: Glog pattern used to collect the shard files list.
        Defaults to "*.tfrec".

        num_repeat: How many times to repeat the dataset. Defaults to -1 (infinite).

    Returns:
        A `TF.data.dataset` ready to be consumed by the model.
    """
    shards_list = [
        i.decode()
        for i in tf.io.matching_files(os.path.join(shard_path, shard_suffix))
        .numpy()
        .tolist()
    ]
    logging.debug(f"found {shards_list}")
    total_shards = len(shards_list)
    logging.info(f"found {total_shards} shards")

    if not prefetch_size:
        prefetch_size = 10

    # how many shard to iterate over in parallels.
    cycle_length = shards_per_cycle if shards_per_cycle else total_shards
    # how many threads to use when fetching inputs from the cycle shards
    num_parallel_calls = cycle_length if async_cycle else 1

    with tf.device("/cpu:0"):
        # shuffle the shard order
        ds = tf.data.Dataset.from_tensor_slices(shards_list)

        # shuffle shard order
        ds = ds.shuffle(total_shards)

        # This is the tricky part, we are using the interleave function to
        # do the sampling as requested by the user. This is not the
        # standard use of the function or an obvious way to do it but
        # its by far the faster and more compatible way to do so
        # we are favoring for once those factors over readability
        # deterministic=False is not an error, it is what allows us to
        # create random batch
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(
                x,
                compression_type=compression,
            ),  # noqa
            cycle_length=cycle_length,
            block_length=example_per_class,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
        ds = ds.map(deserialization_fn, num_parallel_calls=parallelism)
        ds = ds.repeat(count=num_repeat)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(prefetch_size)
        return ds
