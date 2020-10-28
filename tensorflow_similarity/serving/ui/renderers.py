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

from tensorflow_similarity.utils.config_utils import register_custom_object
import numpy as np
from PIL import Image
from io import BytesIO
import base64


class MetadataRenderer(object):
    def __init__(self):
        pass

    def render(self, blob):
        raise NotImplementedError()


class ImageRenderer(object):
    def render(self, decoded_image_bytes):
        """
        decoded_image_bytes - color information for in image, in HWC order
        """
        with BytesIO() as tmp:
            img = Image.fromarray(
                np.array(
                    decoded_image_bytes,
                    dtype=np.uint8))
            img.save(tmp, format="PNG")
            tmp_contents = tmp.getvalue()
            encoded_bytes = base64.b64encode(tmp_contents).decode("ascii")
            return """<img src="data:image/png;base64,%s">""" % encoded_bytes


class Base64ImageRenderer(object):
    def render(self, encoded_bytes):
        return """<img src="data:image/png;base64,%s">""" % encoded_bytes


class TextRenderer(object):
    def render(self, text):
        return text


register_custom_object("Base64ImageRenderer", Base64ImageRenderer)
register_custom_object("ImageRenderer", ImageRenderer)
register_custom_object("TextRenderer", TextRenderer)
