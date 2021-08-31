import tensorflow as tf

from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.callbacks import SplitValidationLoss


def test_eval_callback(tmp_path):
    queries = tf.constant([[1, 2], [1, 2]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1, 2], [1, 2]])
    target_labels = tf.constant([1, 2])

    log_dir = tmp_path / 'sec/'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
            queries=queries,
            query_labels=query_labels,
            targets=targets,
            target_labels=target_labels,
            tb_logdir=str(log_dir)
    )

    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2, ))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile('adam', loss='mse', distance='cosine')
    callback.model = model

    # call the only callback method implemented
    callback.on_epoch_end(0, {})


def test_split_val_loss_callback(tmp_path):
    queries = tf.constant([[1, 2], [1, 2]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1, 2], [1, 2]])
    target_labels = tf.constant([1, 2])
    known_classes = tf.constant([1])

    log_dir = tmp_path / 'sec/'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = SplitValidationLoss(
            queries=queries,
            query_labels=query_labels,
            targets=targets,
            target_labels=target_labels,
            known_classes=known_classes,
            tb_logdir=str(log_dir)
    )

    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2, ))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile('adam', loss='mse', distance='cosine')
    callback.model = model

    # call the only callback method implemented
    callback.on_epoch_end(0, {})
