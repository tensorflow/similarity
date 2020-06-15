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

import shutil

with open("setup.py", "rt") as i:
    with open("setup.py_new", "wt") as o:
        for line in i.readlines():
            tokens = line.split(" ")
            if tokens[0] == '__version__':
                version_str = tokens[-1]
                # Drop the quotes.
                version_str = version_str[1:-2]
                version_tokens = version_str.split(".")
                version_tokens[-1] = str(int(version_tokens[-1]) + 1)
                version_str = ".".join(version_tokens)
                version_str = "'%s'\n" % version_str
                tokens[-1] = version_str
                o.write(' '.join(tokens))
            else:
                o.write(line)

shutil.move("setup.py_new", "setup.py")
