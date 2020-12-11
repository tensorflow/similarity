import tensorflow as tf


def is_tensor_or_variable(x):
    "check if a variable is tf.Tensor or tf.Variable"
    return tf.is_tensor(x) or isinstance(x, tf.Variable)


def tf_cap_memory():
    "Avoid TF to hog memory before needing it"
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
