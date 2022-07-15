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

from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import tensorflow_datasets as tfds
from tqdm.auto import tqdm

from tensorflow_similarity.types import FloatTensor, IntTensor

from .memory_samplers import MultiShotMemorySampler
from .samplers import Augmenter

PreProcessFn = Callable[[FloatTensor, IntTensor], Tuple[FloatTensor, IntTensor]]

T = TypeVar("T", FloatTensor, IntTensor)


class TFDatasetMultiShotMemorySampler(MultiShotMemorySampler):
    def __init__(
        self,
        dataset_name: str,
        classes_per_batch: int,
        x_key: str = "image",
        y_key: str = "label",
        splits: Union[str, Sequence[str]] = ["train", "test"],
        examples_per_class_per_batch: int = 2,
        steps_per_epoch: int = 1000,
        class_list: Sequence[int] = None,
        total_examples_per_class: int = None,
        preprocess_fn: Optional[PreProcessFn] = None,
        augmenter: Optional[Augmenter] = None,
        warmup: int = -1,
    ):
        """Create a Multishot in memory sampler from a dataset downloaded from
        the [TensorFlow datasets catalogue](https://www.tensorflow.org/datasets/catalog/)

        The sampler ensures that each batch is well balanced by ensure that
        each batch aims to contains `example_per_class` examples of
        `classes_per_batch` classes.

        The `batch_size` used during training will be equal to:
        `classes_per_batch * example_per_class` unless an `augmenter` that
        alters the number of examples returned is used. Then the batch_size is
        a function of how many augmented examples are returned by the
        `augmenter`.

        Multishot samplers are to be used when you have multiple examples for
        the same class. If this is not the case, then see the
        [SingleShotMemorySampler()](single_memory.md) for using single example
        with augmentation.

        Memory samplers are good for datasets that fit in memory. If you have
        larger ones that needs to sample from disk then use the
        [TFRecordDatasetSampler()](tfdataset_sampler.md)

        Args:
            dataset_name: the name of the dataset to download and uses as
            referenced in the TensorFlow catalog dataset page.

            x_key: name of the dictonary key that contains the data to feed as
            model input as referenced in the TensorFlow catalog dataset page.
            Defaults to "image".

            y_key: name of the dictonary key that contains the labels as
            referenced in the TensorFlow catalog dataset page.
            Defaults to "label".

            splits: which dataset split(s) to use. Default
            is ["train", "train"] Refersto the catalog page for
            the list of available splits.

            examples_per_class_per_batch: How many example of each class to
            use per batch. Defaults to 2.

            steps_per_epoch: How many steps/batches per epoch.
            Defaults to 1000.

            class_list: Filter the list of examples to only keep those who
            belong to the supplied class list.

            total_examples_per_class: Restrict the number of examples for EACH
            class to total_examples_per_class if set. If not set, all the
            available examples are selected. Defaults to None - no selection.

            preprocess_fn: Preprocess function to apply to the dataset after
            download e.g to resize images. Takes an x and a y.
            Defaults to None.

            augmenter: A function that takes a batch in and return a batch out.
            Can alters the number of examples returned which in turn change the
            batch_size used. Defaults to None.

            warmup: Keep track of warmup epochs and let the augmenter knows
            when the warmup is over by passing along with each batch data a
            boolean `is_warmup`. See `self._get_examples()` Defaults to 0.
        """

        # dealing with users passing a single split e.g "train"
        # instead of ["train"]
        if isinstance(splits, str):
            splits = [splits]

        # we are reusing the memory sampler so "all we need to do" is convert
        # the splits into memory arrays and call the super.
        x = []
        y = []
        for split in splits:
            ds, ds_info = tfds.load(dataset_name, split=split, with_info=True)

            if x_key not in ds_info.features:
                raise ValueError("x_key not found - available features are:", str(ds_info.features.keys()))
            if y_key not in ds_info.features:
                raise ValueError("y_key not found - available features are:", str(ds_info.features.keys()))

            pb = tqdm(total=ds_info.splits[split].num_examples, desc="converting %s" % split)

            for e in ds:
                x.append(e[x_key])
                y.append(e[y_key])
                pb.update()
            pb.close()

        # apply preprocess if needed.
        if preprocess_fn:
            x_pre = []
            y_pre = []
            for idx in tqdm(range(len(x)), desc="Preprocessing data"):
                xb, yb = preprocess_fn(x[idx], y[idx])
                x_pre.append(xb)
                y_pre.append(yb)

            x = x_pre
            y = y_pre

        # delegate to the base memorysample
        super().__init__(
            x,
            y,
            classes_per_batch=classes_per_batch,
            examples_per_class_per_batch=examples_per_class_per_batch,
            steps_per_epoch=steps_per_epoch,
            class_list=class_list,
            total_examples_per_class=total_examples_per_class,
            augmenter=augmenter,
            warmup=warmup,
        )

    def _get_slice(self, input_: T, begin: int, size: int) -> T:
        # x and y are lists of tensors, so we need to use python slicing.
        slice_: T = input_[begin : begin + size]
        return slice_
