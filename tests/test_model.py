import tensorflow as tf
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.models import SimilarityModel


def test_save_and_reload(tmp_path):
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile(optimizer='adam', loss=TripletLoss())

    # index data
    x = tf.constant([[1, 1, 3], [3, 1, 2]], dtype='float32')
    y = tf.constant([1, 2])
    model.index(x, y)

    # save
    model.save(tmp_path)

    # reload
    loaded_model = tf.keras.models.load_model(tmp_path)
    loaded_model.load_index(tmp_path)
    assert loaded_model._index.size() == len(y)


def test_save_no_compile(tmp_path):

    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)

    model.save(tmp_path)
    model2 = tf.keras.models.load_model(tmp_path)
    assert isinstance(model2, type(model))
    

def test_index_single():
    """Unit Test for #161 & #162"""
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile(optimizer='adam', loss=TripletLoss())

    # index data
    x = tf.constant([1, 1, 3], dtype='float32')
    y = tf.constant([1])

    # run individual sample & index
    model.index_single(x, y, data=x)
    assert model._index.size() == 1
