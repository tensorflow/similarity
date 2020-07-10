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
from tensorflow_similarity.serving.www.utils import (load_model)
import tensorflow as tf
import json

class FlaskTestCase(unittest.TestCase):
    
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

    def test_processing_request(self):
        requestDict = {'data': {'0': 0, '1': 1}, 'dataset': 'mnist'}
        # request = json.dumps(requestDict)
        # sprint(request)
        size = 2
        result = processing_request(requestDict, size)
        print(result)
        

if __name__ == "__main__":
    unittest.main()