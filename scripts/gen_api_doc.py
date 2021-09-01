# Lint as: python3
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import tensorflow_privacy
"""Script to generate api_docs for TensorFlow Similarity.

$ pip install git+https://github.com/tensorflow/docs
$ python build_tf_org_api_docs.py --output_dir=/tmp/tfsim
"""
import os
import re
from pathlib import Path
import shutil
from absl import app
from absl import flags

import tensorflow as tf
from termcolor import cprint
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api
from tensorflow_similarity import api as TFSimilarity

tf.config.set_visible_devices([], 'GPU')

OUTDIR = '../api/'

flags.DEFINE_string('output_dir', OUTDIR, 'Where to output the docs.')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity',
    'The url prefix for links to code.')
flags.DEFINE_string('site_path', 'similarity/api_docs/python/',
                    'The location of the doc setin the site.')
flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files.')
flags.DEFINE_bool('gen_report', False,
                  ('Generate an API report containing the health of the'
                   'docstrings of the public API.'))

FLAGS = flags.FLAGS

PROJECT_SHORT_NAME = 'TFSimilarity'
PROJECT_FULL_NAME = 'TensorFlow Similarity'


def replace_in_file(fname, replacements):
    "replace content in file"
    cprint('|-Patching %s' % fname, 'cyan')
    content = open(fname).read()
    os.unlink(fname)

    for rep in replacements:
        content = re.sub(rep[0], rep[1], content, flags=re.MULTILINE)

    # fix the header manually as there is no easy regex
    head = []
    body = []
    is_head = True
    for l in content.split('\n'):

        # fix h3 first
        l = l.replace("><code>", '>')
        l = l.replace("</code></h3>", '</h3>')

        # stop when getting subsection
        if len(l) and l[0] == "##" or "<!-- Placeholder" in l:
            is_head = False

        l = l.replace("<code>", "```python\n")
        l = l.replace('</code>', "```\n")
        l = l.replace('</pre>', '')
        l = l.replace("&#x27;", '')

        if is_head:

            # remove remaining html
            if "on GitHub" in l:
                continue

            if "<" in l:
                continue
            head.append(l)
        else:
            if '<pre' in l:
                continue
            l = re.sub('`([^`]+)`', '<b>\g<1></b>', l)
            l = re.sub('{([^`]+)}', '<i>\g<1></i>', l)

            body.append(l)

    #print(head)

    content = "\n".join(head)
    content += "\n".join(body)

    with open(fname, 'w+') as f:
        f.write(content)


def _hide_layer_and_module_methods():
    """Hide methods and properties defined in the base classes
    of keras layers."""
    # __dict__ only sees attributes defined in *this* class,
    # not on parent classes
    # Needed to ignore redudant subclass documentation
    layer_contents = list(tf.keras.layers.Layer.__dict__.items())
    model_contents = list(tf.keras.Model.__dict__.items())
    module_contents = list(tf.Module.__dict__.items())
    optimizer_contents = list(tf.compat.v1.train.Optimizer.__dict__.items())

    for name, obj in model_contents + layer_contents + module_contents + optimizer_contents:

        if name == '__init__':
            continue

        if isinstance(obj, property):
            obj = obj.fget

        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__

        try:
            doc_controls.do_not_doc_in_subclasses(obj)
        except AttributeError:
            pass


def gen_api_docs():
    """Generates api docs for the tensorflow docs package."""
    output_dir = FLAGS.output_dir

    _hide_layer_and_module_methods()
    doc_generator = generate_lib.DocGenerator(
        root_title=PROJECT_FULL_NAME,
        py_modules=[(PROJECT_SHORT_NAME, TFSimilarity)],
        base_dir=Path(TFSimilarity.__file__).parents[1],
        code_url_prefix=FLAGS.code_url_prefix,
        site_path=FLAGS.site_path,
        search_hints=FLAGS.search_hints,
        # This filter ensures that only objects in init files are only
        # documented if they are explicitly imported. (implicit imports are
        # skipped)
        callbacks=[public_api.explicit_package_contents_filter])

    doc_generator.build(output_dir)


def main(_):
    # cleanup
    outpath = Path(OUTDIR)
    if outpath.exists():
        shutil.rmtree(OUTDIR)
    outpath.mkdir(parents=True)
    cprint('output dir: %s' % OUTDIR, 'green')

    # generate
    gen_api_docs()

    # fixing
    cprint('Fixing documentation', 'magenta')

    cprint("rename main file to md")
    mfname = OUTDIR + 'README.md'
    shutil.move(OUTDIR + 'TFSimilarity.md', mfname)

    reps = [
        [
            "<!-- Insert buttons and diff -->",
            """TensorFlow Similarity is a TensorFlow library focused on making metric learning easy"""
        ],
        [
            "# Module: TFSimilarity",
            "# TensorFlow Similarity API Documentation"
        ],
    ]

    replace_in_file(mfname, reps)

    cprint("[Bulk patching]", 'yellow')
    # pattern, replacement
    reps = [
        ["description: .+", ""],  # remove "pseudo frontmatter"
        ["^\[", '- ['],  # make list valid again
        ["[^#]+\# ", '# '],
        [" module", ''],
    ]

    for fname in Path(OUTDIR).glob('**/*md'):
        fname = str(fname)
        replace_in_file(fname, reps)


if __name__ == '__main__':
    app.run(main)
