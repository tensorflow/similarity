from tests.test_metrics import compute_vector
import tensorflow as tf
from tensorflow.python.keras.saving.save import load_model
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.model import SimilarityModel
from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.distance_metrics import dist_gap, min_neg, max_pos


def generate_dataset(num_classes, num_examples_per_class, reps=4):
    """Generate a dummy datset

    Args:
        num_classes (int): number of class in the dataset.
        num_examples_per_class (int): how many example to generate per class.
        reps (int, optional): How many patterns repetition in X. Defaults to 4.

    Returns:
        list: x, y
    """
    y = []
    x = []
    for i in range(num_classes):
        y.extend([i] * num_examples_per_class)
        vect = [0] * num_classes * 4
        for rep in range(reps):
            idx = i + num_classes * rep
            vect[idx] = 1
        x.extend([vect for _ in range(num_examples_per_class)])
    return tf.constant(x, dtype='float32'), tf.constant(y, dtype='int32')


def test_basic_flow(tmp_path):
    NUM_CLASSES = 8
    REPS = 4
    EXAMPLES_PER_CLASS = 64
    CLASS_PER_BATCH = 8
    BATCH_PER_EPOCH = 500
    BATCH_SIZE = 16
    K = 5
    NUM_MATCHES = 3

    distance = 'cosine'
    positive_mining_strategy = 'hard'
    negative_mining_strategy = 'semi-hard'

    x, y = generate_dataset(NUM_CLASSES, EXAMPLES_PER_CLASS)
    # print(x.shape)
    # print(y.shape)
    # print(x[0])
    # print(y[0])
    # quit()
    sampler = MultiShotMemorySampler(x,
                                    y,
                                    class_per_batch=CLASS_PER_BATCH,
                                    batch_size=BATCH_SIZE,
                                    batch_per_epoch=BATCH_PER_EPOCH)

    # model
    inputs = tf.keras.layers.Input(shape=(NUM_CLASSES * REPS, ))
    # dont use x as variable
    m = tf.keras.layers.Dense(8, activation='relu')(inputs)
    outputs = MetricEmbedding(4)(m)
    model = SimilarityModel(inputs, outputs)

    # loss
    triplet_loss = TripletLoss(distance=distance,
                            positive_mining_strategy=positive_mining_strategy,
                            negative_mining_strategy=negative_mining_strategy)

    # compile
    metrics = [dist_gap(distance), min_neg(distance), max_pos(distance)]
    model.compile(optimizer='adam', metrics=metrics, loss=triplet_loss)

    # train
    history = model.fit(sampler,
                        batch_size=BATCH_SIZE,
                        epochs=1)

    # check that history is properly filled
    assert 'loss' in history.history
    assert 'dist_gap' in history.history

    # indexing ensuring that index is working
    model.reset_index()
    model.index(x, y)
    assert model.index_size() == len(x)

    # # lookup
    neighboors = model.single_lookup(x[0], k=K)
    assert len(neighboors) == K
    # check the model returns reasonable matching
    assert neighboors[0]['label'] == 0

    # check also the last x example which should be for the last class
    neighboors = model.single_lookup(x[-1], k=K)
    assert len(neighboors) == K
    # check the model returns reasonable matching
    assert neighboors[0]['label'] == NUM_CLASSES - 1

    # calibration
    calibration = model.calibrate(x, y, verbose=0)
    assert 'thresholds' in calibration

    # # evaluation
    metrics = model.evaluate_matching(x, y)
    assert 'optimal' in metrics
    assert 0 <= metrics['optimal']['precision'] <= 1
    assert 0 <= metrics['optimal']['recall'] <= 1
    assert 0 <= metrics['optimal']['f1_score'] <= 1

    # matchings
    matches = model.match(x[:NUM_MATCHES], cutpoint='optimal')
    assert len(matches) == NUM_MATCHES

    # # index summary
    model.index_summary()

    # # model save
    model.save(tmp_path)

    # # model load
    mdl2 = load_model(tmp_path)
    mdl2.load_index(tmp_path)
