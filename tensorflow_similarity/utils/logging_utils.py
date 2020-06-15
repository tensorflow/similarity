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
import tensorflow as tf
import logging
import os

LOGGER = None
NAMED_LOGGERS = {}

flags.DEFINE_string("logging_dir", "/tmp/",
                    "Directory in which to write logs.")

FLAGS = flags.FLAGS


def get_logger(id=os.getpid()):
    return setup_logger(id)


def setup_named_logger(name):
    global NAMED_LOGGERS
    if name in NAMED_LOGGERS:
        return NAMED_LOGGERS[name]

    logger = logging.getLogger("moirai_%s" % name)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(pathname)s:%(lineno)s - %(levelname)s - %(message)s')

    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(
        os.path.join(FLAGS.logging_dir, "moirai_%s.log" % name))
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    NAMED_LOGGERS[name] = logger
    return logger


def setup_logger(pid):

    global LOGGER
    if LOGGER:
        return LOGGER

    logger = logging.getLogger("moirai")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(FLAGS.logging_dir, "moirai.log"))
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    #    ch = logging.StreamHandler()
    #    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(pathname)s:%(lineno)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    #    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    #    logger.addHandler(ch)

    LOGGER = logger
    return logger


def info(msg):
    logger = setup_logger(os.getpid())
    logger.info(msg)


def warn(msg):
    logger = setup_logger(os.getpid())
    logger.warn(msg)


def debug(msg):
    logger = setup_logger(os.getpid())
    logger.debug(msg)


def error(msg):
    logger = setup_logger(os.getpid())
    logger.error(msg)
