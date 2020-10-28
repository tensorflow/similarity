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
import logging
import six
from io import BytesIO
import cv2
from PIL import Image
from tensorflow_similarity.utils.config_utils import load_json_configuration_file, json_dict_to_moirai_obj, register_custom_object
from tensorflow_similarity.ui2.renderers import ImageRenderer
from tensorflow_similarity.preprocessing.images import image_file_to_bytes
import base64
from flask import Flask, render_template


class InputHandler(object):
    def __init__(self,
                 parameter_name,
                 render_template_filename,
                 default_value=None):
        self.parameter_name = parameter_name
        self.default_value = default_value
        self.render_template_filename = render_template_filename

    def render(self, request=None, **kwargs):
        if request:
            value = self.get_value(request)
        else:
            value = self.default_value

        return render_template(
            self.render_template_filename,
            parameter_name=self.parameter_name,
            value=value,
            **kwargs)

    def get_value(self, request):
        if self.parameter_name in request.form:
            return request.form[self.parameter_name]
        else:
            return self.default_value


class StringInputHandler(InputHandler):
    def __init__(self, parameter_name, default_value=""):
        super(StringInputHandler, self).__init__(
            parameter_name,
            "widgets/string_input_widget.html",
            default_value=default_value)


class FileInputHandler(InputHandler):
    def __init__(self,
                 parameter_name,
                 render_template_filename="widgets/file_input_widget.html",
                 default_value=None):
        super(FileInputHandler, self).__init__(
            parameter_name,
            render_template_filename,
            default_value=default_value)

    def get_value(self, request):
        if self.parameter_name not in request.files:
            return self.default_value
        return request.files[self.parameter_name].read()


class ImageInputHandler(FileInputHandler):
    def __init__(self,
                 parameter_name,
                 default_value=None):
        super(ImageInputHandler, self).__init__(
            parameter_name,
            "widgets/image_input_widget.html",
            default_value=default_value)

        self.cached_value = None

    def get_value(self, request):

        if self.cached_value is not None:
            return self.cached_value

        if self.parameter_name not in request.files:
            logging.info("%s not in files." % self.parameter_name)
            return self.default_value

        raw_image_contents = BytesIO(request.files[self.parameter_name].read())
        out_img = np.asarray(Image.open(raw_image_contents))

        self.cached_value = out_img[:]
        return out_img

    def render(self, request, **kwargs):
        raw_value = self.get_value(request)

        if raw_value is not None:
            img = Image.fromarray(raw_value)

            with BytesIO() as tmp:
                img.save(tmp, format="PNG")
                tmp_contents = tmp.getvalue()
                logging.info("Contents: %s" % tmp_contents)
                value = base64.b64encode(tmp_contents).decode("ascii")
                rendered_img = ImageRenderer().render(raw_value)
                logging.info(rendered_img)
        else:
            rendered_img = ""

        filename = ""
        if self.parameter_name in request.files:
            filename = request.files[self.parameter_name].filename
            if isinstance(filename, tuple):
                filename = filename[0]
        else:
            filename = ""

        return render_template(
            self.render_template_filename,
            parameter_name=self.parameter_name,
            filename=filename,
            value=rendered_img,
            **kwargs)


def from_config(cfg):
    print(cfg)
    return json_dict_to_moirai_obj(cfg)


register_custom_object("StringInputHandler", StringInputHandler)
register_custom_object("ImageInputHandler", ImageInputHandler)
