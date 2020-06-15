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

from tensorflow_similarity.utils.config_utils import register_custom_object, deserialize_moirai_object


class MoiraiPairFilter(object):
    def keep(self, a, b):
        raise NotImplementedError

    def get_config(self):
        return {'class_name': self.__class__.__name__, 'config': {}}


class KeepAll(MoiraiPairFilter):
    def keep(self, a, b):
        return True


register_custom_object("KeepAll", KeepAll)


class AndPairFilter(MoiraiPairFilter):
    def __init__(self, filters=[]):
        self.filters = [deserialize_moirai_object(x) for x in filters]

    def keep(self, a, b):
        for f in self.filters:
            if not f.keep(a, b):
                return False
        return True


register_custom_object("AndPairFilter", AndPairFilter)


class OrPairFilter(MoiraiPairFilter):
    def __init__(self, filters=[]):
        self.filters = [deserialize_moirai_object(x) for x in filters]

    def keep(self, a, b):
        for f in self.filters:
            if f.keep(a, b):
                return True
        return False


register_custom_object("OrPairFilter", OrPairFilter)
