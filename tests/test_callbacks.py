import tensorflow as tf
from tensorflow_similarity.model import SimilarityModel
from tensorflow_similarity.callbacks import EvalCallback


def test_eval_callback(tmp_path):
    queries = tf.constant([[1, 2], [1, 2]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1, 2], [1, 2]])
    targets_labels = tf.constant([1, 2])

    log_dir = tmp_path / 'sec/'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(queries,
                            query_labels,
                            targets,
                            targets_labels,
                            tb_logdir=str(log_dir))
    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2, ))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile('adam', loss='mse', distance='cosine')
    callback.model = model

    # call the only callback method implemented
    callback.on_epoch_end(0, {})
