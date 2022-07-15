import tensorflow as tf
from termcolor import cprint


def pytest_configure(config):
    tf.config.set_visible_devices([], "GPU")
    cprint("Tensorflow set to CPU", "green")
