import tensorflow as tf
import pytest
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.callbacks import EvalCallback


def generate_dataset(num_classes, num_examples_per_class, reps=4, outputs=1):
    """Generate a dummy datset

    Args:
        num_classes (int): number of class in the dataset.
        num_examples_per_class (int): how many example to generate per class.
        reps (int, optional): How many patterns repetition in X. Defaults to 4.
        heads (int, optional): Number of outputs.

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

    # copy y if neededs
    y = tf.constant(y, dtype='int32')
    if outputs > 1:
        ny = []
        for _ in range(outputs):
            ny.append(y)
        y = ny

    return tf.constant(x, dtype='float32'), y


NUM_CLASSES = 8
REPS = 4
EXAMPLES_PER_CLASS = 64
CLASS_PER_BATCH = NUM_CLASSES
BATCH_PER_EPOCH = 500
BATCH_SIZE = 16
K = 5
NUM_MATCHES = 3
INDEX_HEAD = None  # default


def test_default_multi_output():
    x, y = generate_dataset(NUM_CLASSES, EXAMPLES_PER_CLASS, outputs=2)

    # model
    inputs = tf.keras.layers.Input(shape=(NUM_CLASSES * REPS, ))
    # dont use x as variable
    m = tf.keras.layers.Dense(8, activation='relu')(inputs)
    o1 = MetricEmbedding(4)(m)
    o2 = MetricEmbedding(4)(m)
    model = SimilarityModel(inputs, [o1, o2])

    # loss
    triplet_loss = TripletLoss()

    # compile
    model.compile(optimizer='adam', loss=triplet_loss)

    # callback
    callbacks = [EvalCallback(x, y[0], x, y[0])]

    # train
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=1, callbacks=callbacks)

    assert model._index.embedding_output == 0
    model.index(x, y[0])
    model.single_lookup(x[0])


def test_specified_multi_output():
    EMBEDDING_OUTPUT = 1
    x, y = generate_dataset(NUM_CLASSES, EXAMPLES_PER_CLASS, outputs=2)

    # model
    inputs = tf.keras.layers.Input(shape=(NUM_CLASSES * REPS, ))
    # dont use x as variable
    m = tf.keras.layers.Dense(8, activation='relu')(inputs)
    o1 = MetricEmbedding(6)(m)
    o2 = MetricEmbedding(4)(m)
    model = SimilarityModel(inputs, [o1, o2])

    # loss
    triplet_loss = TripletLoss()

    # compile
    model.compile(optimizer='adam',
                  loss=triplet_loss,
                  embedding_output=EMBEDDING_OUTPUT)

    # train
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=1)

    assert model._index.embedding_output == EMBEDDING_OUTPUT
    model.index(x, y[0])
    model.single_lookup(x[0])


def test_invalid_output_idx():
    inputs = tf.keras.layers.Input(shape=(NUM_CLASSES * REPS, ))
    # dont use x as variable
    m = tf.keras.layers.Dense(8, activation='relu')(inputs)
    o1 = tf.keras.layers.Dense(6)(m)
    o2 = tf.keras.layers.Dense(4)(m)
    model = SimilarityModel(inputs, [o1, o2])

    # check that specificing an invalid output value raise a valueerror
    with pytest.raises(ValueError):
        model.compile(optimizer='adam', loss='mse', embedding_output=42)
