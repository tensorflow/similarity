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

__MODELS = {}


def __get_name(fn):
    if hasattr(fn, "__name__"):
        return fn.__name__
    elif hasattr(fn, "name"):
        if callable(fn.name):
            return fn.name()
        else:
            return fn.name
    else:
        return fn.__class__.__name__


def register_model(fn, name=None):
    if not name:
        name = __get_name(fn)
    __MODELS[name] = fn


def get_model_names():
    return __MODELS.keys()


def get_model(name):
    return __MODELS[name]
