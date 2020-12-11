from termcolor import cprint
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
cprint('Tensorflow set to CPU', 'green')
