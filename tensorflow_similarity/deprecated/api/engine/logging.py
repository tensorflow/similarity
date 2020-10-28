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

import logging
import os

LOGGER = None


def get_logger():
    global LOGGER
    if not LOGGER:
        LOGGER = setup_logger()
    return LOGGER


def setup_logger():
    global LOGGER

    logger = logging.getLogger("moirai")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("moirai.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(pathname)s:%(lineno)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    LOGGER = logger
    return logger


def info(msg):
    global LOGGER
    if not LOGGER:
        LOGGER = setup_logger()
    LOGGER.info(msg)


def warning(msg):
    global LOGGER
    if not LOGGER:
        LOGGER = setup_logger()
    LOGGER.warning(msg)


def debug(msg):
    global LOGGER
    if not LOGGER:
        LOGGER = setup_logger()
    LOGGER.debug(msg)


def error(msg):
    global LOGGER
    if not LOGGER:
        LOGGER = setup_logger()
    LOGGER.error(msg)
