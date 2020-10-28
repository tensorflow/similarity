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

from absl import app, flags
import h5py
import json
import numpy as np
from PIL import Image
from io import BytesIO
import base64

flags.DEFINE_string("input_file", "icons.h5", "Name of the input file.")
flags.DEFINE_string("output_file", "icons_w_metadata.h5",
                    "Name of the input file.")
flags.DEFINE_string("image_field", "logos", "Name of the training.")

FLAGS = flags.FLAGS


def main(args):
    i = h5py.File(FLAGS.input_file, "r")
    o = h5py.File(FLAGS.output_file, "w")

    # Copy datasets.
    for dataset_name in i:
        o.create_dataset(dataset_name, data=i[dataset_name][:])

    str_dtype = h5py.special_dtype(vlen=str)
    metadata = []
    for item in i[FLAGS.image_field]:
        with BytesIO() as tmp:
            img = Image.fromarray(np.array(item, dtype=np.uint8))
            img = img.resize((32, 32))
            img.save(tmp, format="PNG")
            tmp_contents = tmp.getvalue()
            encoded_bytes = base64.b64encode(tmp_contents).decode("ascii")

            metadatum = {
                'display_renderer': 'Base64ImageRenderer',
                'display_data': encoded_bytes
            }
            json_metadatum = json.dumps(metadatum)
            metadata.append(json_metadatum)

    o.create_dataset("metadata", shape=(len(metadata),), dtype=str_dtype)
    o["metadata"][:] = metadata

    i.close()
    o.close()


if __name__ == '__main__':
    app.run(main)
