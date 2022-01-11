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


import abc
from tensorflow_similarity.types import Tensor
from typing import List


class Augmenter(abc.ABC):

    @abc.abstractmethod
    def augment(self, x: Tensor, y: Tensor, num_augmentations_per_example: int,
                is_warmup: bool) -> List[Tensor]:
        pass

    def __call__(self, x: Tensor, y: Tensor,
                 num_augmentations_per_example: int,
                 is_warmup: bool) -> List[Tensor]:
        return self.augment(x, y, num_augmentations_per_example, is_warmup)
