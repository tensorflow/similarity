import pytest
import tensorflow as tf
from termcolor import cprint

from tensorflow_similarity.losses import SimSiamLoss, TripletLoss
from tensorflow_similarity.models import ContrastiveModel, SimilarityModel
from tensorflow_similarity.models.contrastive_model import load_model


@pytest.fixture
def use_gpu_if_available():
    """Override config in `conftest.py` to use GPU if available."""
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 0:
        cprint("Tensorflow set to use GPU", "green")
        with tf.device("/gpu:0"):
            yield
    else:
        yield


def test_save_and_reload(tmp_path):
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile(optimizer="adam", loss=TripletLoss())

    # index data
    x = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="float32")
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
    model.compile(optimizer="adam", loss=TripletLoss())

    # index data
    x = tf.constant([1, 1, 3], dtype="float32")
    y = tf.constant([1])

    # run individual sample & index
    model.index_single(x, y, data=x)
    assert model._index.size() == 1


@pytest.mark.usefixtures("use_gpu_if_available")
def test_save_and_reload_contrastive_model(tmp_path):
    """Test save and load of ContrastiveModel.

    Testing it also in a MirroredStrategy on GPU if available, to check fix for
    issue #287.
    """
    with tf.distribute.MirroredStrategy().scope():
        backbone_input = tf.keras.layers.Input(shape=(3,))
        backbone_output = tf.keras.layers.Dense(4)(backbone_input)
        backbone = tf.keras.Model(
            inputs=backbone_input,
            outputs=backbone_output,
        )

        projector_input = tf.keras.layers.Input(shape=(4,))
        projector_output = tf.keras.layers.Dense(4)(projector_input)
        projector = tf.keras.Model(
            inputs=projector_input,
            outputs=projector_output,
        )

        predictor_input = tf.keras.layers.Input(shape=(4,))
        predictor_output = tf.keras.layers.Dense(4)(predictor_input)
        predictor = tf.keras.Model(
            inputs=predictor_input,
            outputs=predictor_output,
        )

        model = ContrastiveModel(
            backbone=backbone,
            projector=projector,
            predictor=predictor,
        )
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.5)

        model.compile(optimizer=opt, loss=SimSiamLoss())

    # test data
    x = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="int64")

    # create dataset with two views
    ds = tf.data.Dataset.from_tensors(x)
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)

    model.fit(ds)

    # save
    model.save(tmp_path)

    # reload and test loaded model
    with tf.distribute.MirroredStrategy().scope():
        loaded_model = load_model(tmp_path)

    pred = loaded_model.predict(x)

    assert loaded_model.algorithm == "simsiam"
    assert loaded_model.optimizer.lr == 0.5
    assert loaded_model.backbone.input_shape == (None, 3)
    assert loaded_model.backbone.output_shape == (None, 4)
    assert loaded_model.predictor.input_shape == (None, 4)
    assert loaded_model.predictor.output_shape == (None, 4)
    assert loaded_model.projector.input_shape == (None, 4)
    assert loaded_model.projector.output_shape == (None, 4)
    assert pred.shape == (2, 4)
