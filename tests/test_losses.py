import tensorflow as tf
import numpy as np
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.losses import PNLoss


# [triplet loss]
def test_triplet_loss_serialization():
    loss = TripletLoss()
    config = loss.get_config()
    print(config)
    loss2 = TripletLoss.from_config(config)
    assert loss.name == loss2.name
    assert loss.distance == loss2.distance


def triplet_hard_loss_np(labels, embedding, margin, dist_func, soft=False):

    num_data = embedding.shape[0]
    # Reshape labels to compute adjacency matrix.
    labels_reshaped = np.reshape(labels.astype(np.float32),
                                 (labels.shape[0], 1))

    adjacency = np.equal(labels_reshaped, labels_reshaped.T)
    pdist_matrix = dist_func(embedding)
    loss_np = 0.0
    for i in range(num_data):
        pos_distances = []
        neg_distances = []
        for j in range(num_data):
            if adjacency[i][j] == 0:
                neg_distances.append(pdist_matrix[i][j])
            if adjacency[i][j] > 0.0 and i != j:
                pos_distances.append(pdist_matrix[i][j])

        # if there are no positive pairs, distance is 0
        if len(pos_distances) == 0:
            pos_distances.append(0)

        # Sort by distance.
        neg_distances.sort()
        min_neg_distance = neg_distances[0]
        pos_distances.sort(reverse=True)
        max_pos_distance = pos_distances[0]

        if soft:
            loss_np += np.log1p(np.exp(max_pos_distance - min_neg_distance))
        else:
            loss_np += np.maximum(0.0,
                                  max_pos_distance - min_neg_distance + margin)

    loss_np /= num_data
    return loss_np


def test_triplet_loss():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs, ), 0, 10, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 20), 0, 1)
    tpl = TripletLoss()
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss > 0.9


def test_triplet_loss_easy():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs, ), 0, 3, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 16), 0, 1)
    tpl = TripletLoss(positive_mining_strategy='easy',
                      negative_mining_strategy='easy')
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss > 0


def test_triplet_loss_semi_hard():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs, ), 0, 3, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 16), 0, 1)
    tpl = TripletLoss(positive_mining_strategy='easy',
                      negative_mining_strategy='semi-hard')
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss


def test_triplet_loss_hard():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs, ), 0, 3, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 16), 0, 1)
    tpl = TripletLoss(positive_mining_strategy='hard',
                      negative_mining_strategy='hard')
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss


# [pn loss]
def test_pn_loss_serialization():
    loss = PNLoss()
    config = loss.get_config()
    print(config)
    loss2 = PNLoss.from_config(config)
    assert loss.name == loss2.name
    assert loss.distance == loss2.distance


def test_np_loss():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs, ), 0, 10, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 20), 0, 1)
    pnl = PNLoss()
    # y_true, y_preds
    loss = pnl(y_true, y_preds)
    assert loss > 0.9
