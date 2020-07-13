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
from tensorflow_similarity.serving.www.utils import (load_model, get_imdb_dict, encode_review, decode_review)
import numpy as np
import json
import requests

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
        #print(encoded_arr)
        test_arr = np.asarray([1,14,20,16,777,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.assertTrue((encoded_arr == test_arr).all())

    def test_decode_review(self):
        decoded_text = decode_review(np.asarray([1,14,20,16,777,0]))
        self.assertEqual("<START> this movie was fantastic <PAD>", decoded_text)

    def test_mnist_response(self):
        headers = headers = {'Content-Type': 'application/json' }
        payload = {'data': {'0': 0, '1': 1}, 'dataset': 'mnist'}
        url = "http://127.0.0.1:5000/distances"
        response = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))
        self.assertEqual(response.json()['predicted_label'], '3')
        self.assertEqual(response.status_code, 200)

    def test_imdb_response(self):
        headers = headers = {'Content-Type': 'application/json' }
        payload = {'data': "This movie sucked", 'dataset': 'imdb'} 
        url = "http://127.0.0.1:5000/distances"
        response = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))
        self.assertEqual(response.json()['predicted_label'], '0')
        self.assertEqual(response.status_code, 200)

    def test_emoji_response(self):
        headers = headers = {'Content-Type': 'application/json' }
        payload = {'data': {'0': 0, '1': 1}, 'dataset': 'emoji'}
        url = "http://127.0.0.1:5000/distances"
        response = requests.post(url, headers=headers, data=json.dumps(payload,indent=4))
        self.assertEqual(response.json()['predicted_label'], 'face_without_mouth')
        self.assertEqual(response.status_code, 200)
        

if __name__ == "__main__":
    unittest.main()