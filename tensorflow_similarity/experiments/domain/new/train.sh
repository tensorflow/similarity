#!/bin/sh

# Copyright 2020 Google LLC
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

#python  train.py --api_key c0d3e237a1996ccd --generator_workers 40 --hypertune --sample_fraction 0 --epoch_budget=2000000 --epochs_per_model=20000 --step_size_multiplier .01 --strategy stable_quadruplet_loss --model=bogo


# nohup python train.py \
#       --api_key c0d3e237a1996ccd \
#       --generator_workers 40 \
# #      --hypertune \
#       --sample_fraction 1 \
#       --epoch_budget=500 \
#       --epochs_per_model=5 \
#       --step_size_multiplier .01 \
#       --strategy stable_quadruplet_loss \
#       --model=trigrams & 


nohup python train.py \
      --generator_workers 12 \
      --sample_fraction 1 \
      --epochs=500 \
      --step_size_multiplier 1 \
      --strategy stable_hard_quadruplet_loss \
      --model=trigrams & 
