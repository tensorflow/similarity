import tensorflow as tf
from tensorflow.python.keras.saving.save import load_model
import tensorflow_similarity as tfsim
from tensorflow_similarity.utils import tf_cap_memory
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.model import SimilarityModel
from tensorflow_similarity.sampler import MultiShotMemorySampler
from tensorflow_similarity.sampler import select_examples
from tensorflow_similarity.visualization import viz_neigbors_imgs
from tensorflow_similarity.visualization import confusion_matrix


def test_basic_flow(tmp_path):
    NUM_EXAMPLES = 128
    CLASS_PER_BATCH = 3
    BATCH_SIZE = 16
    K = 5
    NUM_MATCHES = 3


    distance = 'cosine'
    positive_mining_strategy = 'hard'
    negative_mining_strategy = 'semi-hard'

    x = tf.random.normal((NUM_EXAMPLES, 8))
    y = tf.random.uniform((NUM_EXAMPLES, ), minval=0, maxval=10)
    sampler = MultiShotMemorySampler(x,
                                     y,
                                     class_per_batch=CLASS_PER_BATCH,
                                     batch_size=BATCH_SIZE)

    # model
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(8, ))
    outputs = MetricEmbedding(2)(inputs)
    model = SimilarityModel(inputs, outputs)

    # loss
    triplet_loss = TripletLoss(
        distance=distance,
        positive_mining_strategy=positive_mining_strategy,
        negative_mining_strategy=negative_mining_strategy)

    # compile
    model.compile(optimizer='adam', loss=triplet_loss)

    history = model.fit(sampler,
                        validation_data=(x, y),
                        batch_size=BATCH_SIZE,
                        epochs=1)

    assert 'loss' in history.history
    assert 'val_loss' in history.history

    # indexing
    model.reset_index()
    model.index(x, y)

    # lookup
    neighboors = model.single_lookup(x[0], k=K)
    assert len(neighboors) == K

    # calibration
    calibration = model.calibrate(x, y, verbose=0)
    assert 'thresholds' in calibration

    # evaluation
    metrics = model.evaluate_index(x, y)
    # FIXME asserts here

    # matchings
    matches = model.match(x[:NUM_MATCHES], cutpoint='optimal')
    assert len(matches) == NUM_MATCHES

    # index summary
    model.index_summary()

    # model save
    model.save(tmp_path)

    # model load
    mdl2 = load_model(tmp_path)
    mdl2.load_index(tmp_path)
