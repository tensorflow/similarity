import numpy as np
import tensorflow as tf

from tensorflow_similarity import losses


# [pn loss]
def test_pn_loss_serialization():
    loss = losses.PNLoss()
    config = loss.get_config()
    print(config)
    loss2 = losses.PNLoss.from_config(config)
    assert loss.name == loss2.name
    assert loss.distance == loss2.distance


def test_np_loss():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs,), 0, 10, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 20), 0, 1)
    pnl = losses.PNLoss()
    # y_true, y_preds
    loss = pnl(y_true, y_preds)
    assert loss > 0.9


# [soft neasrest neighbor loss]
def test_softnn_loss_serialization():
    loss = losses.SoftNearestNeighborLoss(distance="cosine", temperature=50)
    config = loss.get_config()
    loss2 = losses.SoftNearestNeighborLoss.from_config(config)
    assert loss.name == loss2.name
    assert loss.distance == loss2.distance
    assert loss.temperature == loss2.temperature


def softnn_util(y_true, x, temperature=1):
    """
    A simple loop based implementation of soft
    nearest neighbor loss to test the code.
    https://arxiv.org/pdf/1902.01889.pdf
    """

    y_true = y_true.numpy()
    x = x.numpy()
    batch_size = y_true.shape[0]
    loss = 0
    for i in range(batch_size):
        numerator = 0
        denominator = 0
        for j in range(batch_size):
            if i == j:
                continue
            if y_true[i] == y_true[j]:
                numerator += np.exp(-1 * np.sum(np.square(x[i] - x[j])) / temperature)
            denominator += np.exp(-1 * np.sum(np.square(x[i] - x[j])) / temperature)
        if numerator == 0:
            continue
        loss += np.log(numerator / denominator)
    return -loss / batch_size


def test_softnn_loss():
    num_inputs = 10
    n_classes = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs,), 0, n_classes, dtype=tf.int32)
    # x: embeddings
    x = tf.random.uniform((num_inputs, 20), 0, 1)

    temperature = np.random.uniform(0.1, 50)
    softnn = losses.SoftNearestNeighborLoss(temperature=temperature)
    loss = softnn(y_true, x)
    loss_check = softnn_util(y_true, x, temperature)
    loss_diff = loss.numpy() - loss_check
    assert np.abs(loss_diff) < 1e-3


def test_xbm_loss():
    batch_size = 6
    embed_dim = 16

    embeddings1 = tf.random.uniform(shape=[batch_size, embed_dim])
    labels1 = tf.constant(
        [
            [1],
            [1],
            [2],
            [2],
            [3],
            [3],
        ],
        dtype=tf.int32,
    )

    embeddings2 = tf.random.uniform(shape=[batch_size, embed_dim])
    labels2 = tf.constant(
        [
            [4],
            [4],
            [5],
            [5],
            [6],
            [6],
        ],
        dtype=tf.int32,
    )

    distance = "cosine"
    loss = losses.MultiSimilarityLoss(distance=distance)
    loss_nowarm = losses.XBM(loss, memory_size=12, warmup_steps=0)

    # test enqueue
    loss_nowarm(labels1, embeddings1)
    assert loss_nowarm._y_pred_memory.numpy().shape == (batch_size, embed_dim)
    tf.assert_equal(loss_nowarm._y_true_memory, labels1)

    loss_nowarm(labels2, embeddings2)
    assert loss_nowarm._y_pred_memory.numpy().shape == (
        2 * batch_size,
        embed_dim,
    )
    tf.assert_equal(loss_nowarm._y_true_memory, tf.concat([labels2, labels1], axis=0))

    # test dequeue
    loss_nowarm(labels2, embeddings2)
    assert loss_nowarm._y_pred_memory.numpy().shape == (
        2 * batch_size,
        embed_dim,
    )
    tf.assert_equal(loss_nowarm._y_true_memory, tf.concat([labels2, labels2], axis=0))

    # test warmup
    loss_warm = losses.XBM(loss, memory_size=12, warmup_steps=1)

    loss_warm(labels1, embeddings1)
    assert loss_warm._y_pred_memory.numpy().shape == (0, embed_dim)
    tf.assert_equal(loss_warm._y_true_memory, tf.constant([[]], dtype=tf.int32))

    loss_warm(labels2, embeddings2)
    assert loss_warm._y_pred_memory.numpy().shape == (batch_size, embed_dim)
    tf.assert_equal(loss_warm._y_true_memory, labels2)


# [multiple negatives ranking loss]
def test_multineg_rank_loss_serialization():
    loss = losses.MultiNegativesRankLoss(distance="inner_product")
    config = loss.get_config()
    loss2 = losses.MultiNegativesRankLoss.from_config(config)
    assert loss.name == loss2.name
    assert loss.distance == loss2.distance
