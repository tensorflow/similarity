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


import unittest
from tensorflow_similarity.serving.www.main import (app, processing_request)
from tensorflow_similarity.serving.www.explain import Explainer
from tensorflow_similarity.serving.www.utils import (load_model, get_imdb_dict, encode_review, decode_review, read_image_dataset_targets, read_text_dataset_targets)
from tensorflow_similarity.serving.www.constants import IMDB_REVIEW_LENGTH
import numpy as np
import json
import requests

BASE_TARGET_PATH = "../../../tensorflow_similarity/serving/www/static/"

class ServingTestCase(unittest.TestCase):
    
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get("/")
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)

    def test_load_model(self):
        model, dict_key, explainer = load_model('../../../tensorflow_similarity/serving/www/saved_models/mnist_model.h5')
        self.assertEqual(dict_key, "example")
        self.assertIsNotNone(model)
        self.assertIsInstance(explainer, Explainer)

    def test_get_imdb_dict(self):
        word_index = get_imdb_dict()
        self.assertIsInstance(word_index, dict)
        self.assertEqual(word_index["<PAD>"], 0)
        self.assertEqual(word_index["<START>"], 1)
        self.assertEqual(word_index["<UNK>"], 2)
        self.assertEqual(word_index["<UNUSED>"], 3)
        self.assertEqual(word_index["movie"], 20)

    def test_encode_review(self):
        encoded_arr = encode_review("This movie was fantastic")
        expected_encoding = [1,14,20,16,777]
        while len(expected_encoding) < IMDB_REVIEW_LENGTH:
            expected_encoding.append(0)
        expected_encoding = np.asarray(expected_encoding)
        self.assertTrue((encoded_arr == expected_encoding).all())

    def test_decode_review(self):
        decoded_text = decode_review(np.asarray([1,14,20,16,777,0]))
        self.assertEqual("<START> this movie was fantastic <PAD>", decoded_text)

    def test_read_text_dataset_targets(self):
        imdb_targets, imdb_targets_labels = read_text_dataset_targets(BASE_TARGET_PATH + "/text/imdb_targets/", "text")
        self.assertEqual(len(imdb_targets["text"]), 6)
        self.assertEqual(len(imdb_targets_labels), 6)
        self.assertEqual(imdb_targets_labels, ['0', '0', '0', '1', '1', '1'])

    def test_read_image_dataset_targets(self):
        mnist_targets, mnist_target_labels = read_image_dataset_targets(BASE_TARGET_PATH + "images/mnist_targets/", False, 28, "example")
        emoji_targets, emoji_targets_labels = read_image_dataset_targets(BASE_TARGET_PATH + "/images/emoji_targets/", True, 32, "image")
        self.assertEqual(len(mnist_targets["example"]), 10)
        self.assertEqual(len(mnist_target_labels), 10)
        self.assertTrue((mnist_target_labels == ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']).all())
        self.assertEqual(len(emoji_targets["image"]), 109)
        self.assertEqual(len(emoji_targets_labels), 109)

    def test_mnist_response(self):
        tester = app.test_client(self)
        payload = {'data': {'0': 0, '1': 1}, 'dataset': 'mnist'}
        response = tester.post("/distances", data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.get_json()['predicted_label'], '3')
        self.assertEqual(response.status_code, 200)

    def test_imdb_response(self):
        tester = app.test_client(self)
        payload = {'data': "This movie sucked", 'dataset': 'imdb'} 
        response = tester.post("/distances", data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.get_json()['predicted_label'], '0')
        self.assertEqual(response.status_code, 200)

    def test_emoji_response(self):
        tester = app.test_client(self)
        payload = {'data': {'0': 0, '1': 1}, 'dataset': 'emoji'}
        response = tester.post("/distances", data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.get_json()['predicted_label'], 'face_without_mouth')
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()