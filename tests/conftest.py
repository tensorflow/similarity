from termcolor import cprint
import tensorflow as tf


def pytest_configure(config):
    tf.config.set_visible_devices([], 'GPU')
    cprint('Tensorflow set to CPU', 'green')
