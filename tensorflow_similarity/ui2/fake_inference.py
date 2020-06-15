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

import numpy as np
from flask import render_template, jsonify
from collections import namedtuple, defaultdict
from tensorflow_similarity.dataset import Dataset
from tensorflow_similarity.api.engine.inference import InferenceRequest
import os
import six
from tensorflow_similarity.utils.config_utils import json_dict_to_moirai_obj
from tensorflow_similarity.ui2.input_handler import ImageInputHandler, StringInputHandler, from_config
from tensorflow_similarity.preprocessing.identity import Identity
from tensorflow_similarity.api.engine.inference import InferenceRequest

import traceback
import tensorflow as tf
from PIL import Image


class FakeInference(object):
    def __init__(self,
                 include_precisions=False,
                 include_display_text=False,
                 include_display_img=False):
        self.include_precisions = include_precisions
        self.include_display_text = include_display_text
        self.include_display_img = include_display_img

    def neighbors(self, examples, request=InferenceRequest()):
        response = []

        for i in range(len(examples)):
            element_response = defaultdict(list)
            if request.include_examples:
                element_response['example'] = example

            el_emb = np.array([i, i + 10, i + 20, i + 30])
            if request.include_input_embeddings:
                element_response['embedding'] = el_emb

            for j in range(request.num_neighbors):
                targ_emb = [i + j, i + 10 + j, i + 20 + j, i + 30 + j]
                element_response['labels'].append("label_%d_%d" % (i, j))
                element_response['distances'].append(.125 * j + .05)
                if self.include_precisions:
                    element_response['precisions'].append(1 - (.125 * j + .05))

                if self.include_display_text:
                    element_response['metadata'].append({
                        'display_data': "text_%d" % i,
                        'display_renderer': "TextRenderer"})

                elif self.include_display_img:
                    fake_img = np.ones((32, 32, 3)) * 32 * j
                    fake_img = fake_img.astype(np.uint8)

                    element_response['metadata'].append({
                        'display_data': fake_img,
                        'display_renderer': "ImageRenderer"})
                else:
                    element_response['metadata'].append({
                        'display_data': "default_label_data_%d" % j,
                        'display_renderer': "TextRenderer"})

                if request.include_target_embeddings:
                    element_response['target_embeddings'].append(targ_emb)
                    element_response['target_distance'].append(
                        np.linalg.norm(
                            el_emb - targ_emb, ord=2))
            response.append(element_response)
        return response
