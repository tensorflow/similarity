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

from tensorflow_similarity.api.generators.task_based_generator import TaskBasedGenerator
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils import model_utils
from tensorflow_similarity.api.losses.no_loss import NoLoss
from tensorflow.keras.models import Model
import tensorflow as tf

from tensorflow_similarity.api.engine.task import MainTask, MetaTask
from tensorflow_similarity.api.generators.prewarm import PrewarmGeneratorWrapper


class PrewarmTask(MainTask):
    def __init__(self, main_task):
        name = "%s_prewarm" % main_task.name
        self.main_task = main_task
        generator = main_task.get_generator()
        wrapped_generator = PrewarmGeneratorWrapper(generator)

        super(PrewarmTask, self).__init__(name, wrapped_generator,
                                          main_task.tower_model)

    def build_task(self):
        self.main_task.build(compile=False)
        self.suppressed_metric_prefixes.extend(
            self.main_task.suppressed_metric_prefixes)

        inputs, names = model_utils.clone_task_inputs(self.main_task)
        for n, i in zip(names, inputs):
            self._add_input(n, i)
            out_name = "%s_prewarm_out" % n
            o = Rename(name=out_name)(i)
            self._add_output(out_name, o, NoLoss())

        self.task_model = Model(inputs=self.task_inputs,
                                outputs=self.task_outputs)

    @classmethod
    def for_task(cls, task):
        return cls(task)


def create_prewarming_task(meta_task):
    assert isinstance(meta_task, MetaTask)
    meta_task.main_task.build(compile=False)
    prewarm_task = PrewarmTask(meta_task.main_task)
    prewarm_task.build(compile=False)

    return MetaTask("prewarm_metatask",
                    meta_task.main_task.tower_model,
                    main_task=prewarm_task,
                    auxillary_tasks=meta_task.auxillary_tasks,
                    inference_task=meta_task.inference_task,
                    optimizer=meta_task.optimizer)
