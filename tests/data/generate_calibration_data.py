import numpy as np
import tensorflow as tf
import tensorflow_similarity as tfsim
from tensorflow.keras import layers
from tensorflow_similarity.utils import tf_cap_memory
from tensorflow_similarity.losses import TripletLoss  # specialized loss
from tensorflow_similarity.layers import MetricEmbedding  # layer with l2 regularization
from tensorflow_similarity.model import SimilarityModel  # TF model with additional features
from tensorflow_similarity.distance_metrics import avg_neg, avg_pos, dist_gap, max_pos, min_neg  # various metrics on how the distance between examples evolve.
from tensorflow_similarity.callbacks import EvalCallback  # evaluate matching performance
from tensorflow_similarity.samplers import MultiShotMemorySampler  # sample data
from tensorflow_similarity.samplers import select_examples  # select n example per class
from tensorflow_similarity.visualization import viz_neigbors_imgs  # Neigboors vizualisation
from tensorflow_similarity.visualization import confusion_matrix  # matching performance

FNAME = 'mnist_fashion_embeddings.npz'
EPOCHS = 1
BATCH_SIZE = 32
CLASS_PER_BATCH = 4
BATCH_PER_EPOCH = 1000
DATASET_SIZE = 1000  # non need to go too big

distance = 'cosine'
positive_mining_strategy = 'hard'
negative_mining_strategy = 'semi-hard'

(x_train, y_train), (x_test,
                     y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = tf.constant(x_train / 255.0, dtype='float32')
x_test = tf.constant(x_test / 255.0, dtype='float32')

x_restricted, y_restricted = select_examples(x_train, y_train, list(range(6)), 30000)
sampler = MultiShotMemorySampler(x_restricted,
                                 y_restricted,
                                 classes_per_batch=CLASS_PER_BATCH,
                                 batch_size=BATCH_SIZE,
                                 steps_per_epoch=BATCH_PER_EPOCH)


def get_model():
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(28, 28))
    x = layers.Reshape((28, 28, 1))(inputs)
    x = layers.Conv2D(128, 7, activation='relu')(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    # dont make the embedding to large - its slow down the lookups
    outputs = MetricEmbedding(32)(x)
    return SimilarityModel(inputs, outputs)


model = get_model()

# loss
triplet_loss = TripletLoss(distance=distance,
                           positive_mining_strategy=positive_mining_strategy,
                           negative_mining_strategy=negative_mining_strategy)

# compile
model.compile(optimizer='adam', metrics=[], loss=triplet_loss)

# train
history = model.fit(sampler, batch_size=BATCH_SIZE, epochs=EPOCHS)


x_idx, y_idx = select_examples(x_train, y_train, range(10), DATASET_SIZE//10)
x_cal, y_cal = select_examples(x_test, y_test, range(10), DATASET_SIZE//10)

# compute
embeddings_idx = model.predict(x_idx)
embeddings_cal = model.predict(x_cal)
# store
np.savez(FNAME,
         x_idx=x_idx,
         y_idx=y_idx,
         embeddings_idx=embeddings_idx,
         x_cal=x_cal,
         y_cal=y_cal,
         embeddings_cal=embeddings_cal)
