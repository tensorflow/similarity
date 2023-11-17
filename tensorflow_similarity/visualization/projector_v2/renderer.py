# Copyright 2021 The TensorFlow Authors
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

import abc
import functools
import json
import random
import string
import threading
from http import server
from os import path
from typing import Any, Dict, Optional, Sequence, Union


class Renderer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def display(self):
        pass

    @abc.abstractmethod
    def send_message(
        self,
        cell_id: str,
        msg_type: str,
        payload: Sequence[Any],
        other_payload: Optional[Dict[str, Union[str, int, float]]],
    ):
        pass


class IPythonRenderer(Renderer):
    class _IPythonRequestHandler(server.BaseHTTPRequestHandler):
        def __init__(self, *args, callback=None, **kwargs):
            self._cb = callback
            super(IPythonRenderer._IPythonRequestHandler, self).__init__(
                *args, **kwargs
            )

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Request-Method", "POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write("OK".encode("utf8"))

        def do_POST(self):
            if self._cb:
                content_length = int(self.headers["Content-Length"])
                form_content = self.rfile.read(content_length)
                payload = json.loads(form_content)

                self._cb(payload)

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write("OK".encode("utf8"))

    def __init__(self):
        self._server = server.ThreadingHTTPServer(
            ("", 0),
            functools.partial(
                IPythonRenderer._IPythonRequestHandler, callback=self._on_req
            ),
        )
        thread = threading.Thread(
            target=self._server.serve_forever,
        )
        thread.daemon = True
        thread.start()

    def _generate_id(self):
        return "".join(random.choices(string.ascii_letters, k=12))

    def display(self):
        from IPython import display

        display.clear_output()
        with open(path.join(path.dirname(__file__), "bin", "index.js"), "r") as f:
            library = display.Javascript(f.read())

        unique_id = self._generate_id()
        container = display.HTML(
            f'<div id="{unique_id}" style="height: 100%; width: 100%;"></div>'
        )
        bootstrap = display.Javascript(
            """
            globalThis.messenger.initForIPython(%d);
            globalThis.messenger.createMessengerForOutputcell("%s");
            globalThis.bootstrap("%s");
        """
            % (self._server.server_port, unique_id, unique_id)
        )
        display.display(
            library,
            container,
        )
        display.display(
            bootstrap,
        )

        return unique_id

    def send_message(
        self,
        cell_id: str,
        msg_type: str,
        payload: Sequence[Any],
        other_payload: Optional[Dict[str, Union[str, int, float]]] = {},
    ):
        from IPython import display

        unique_id = self._generate_id()
        send_payload = display.Javascript(
            f"""
            globalThis["{unique_id}"] = {json.dumps(payload)};
        """
        )
        load_payload = display.Javascript(
            f"""
            globalThis.messenger.onMessageFromPython(
              "{cell_id}",
              "{msg_type}",
              ["{unique_id}"],
              {json.dumps(other_payload)}
            );
        """
        )
        display.display(send_payload)
        display.display(load_payload)
