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

from __future__ import absolute_import

import argparse
import logging
import re
import os

from past.builtins import unicode

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import json
from tensorflow_similarity.benchmark import read_config_file, run_experiment


def run(argv=None):
    """Main entry point; defines and runs the tensorflow similarity benchmark
       pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        default='example_image_config.json',
        help='Input file to process.')
    parser.add_argument('--output',
                        dest='output',
                        default='example_output.txt',
                        help='Output file to write results to.')
    parser.add_argument('--runner',
                        dest='runner',
                        default='DirectRunner',
                        help='Specify where we want to execute the pipeline.')
    parser.add_argument('--job_name',
                        dest='job_name',
                        default='tf_similarity_benchmark_job',
                        help='The job name for the pipeline execution.')

    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        '--runner={}'.format(known_args.runner),
        '--job_name={}'.format(known_args.job_name),
    ])

    path = os.path.abspath(known_args.input)
    # read the config file into a list of configs
    configs_lst = read_config_file(path)

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module
    # level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:

        # Transform the list of configs into a PCollection.
        configs = p | beam.Create(configs_lst)

        # run experiment for each config
        outputs = configs | beam.Map(run_experiment)

        # Write the output using a "Write" transform that has side effects.
        # pylint: disable=expression-not-assigned
        outputs | WriteToText(known_args.output)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
