import tensorflow as tf
from tensorflow.keras.losses import Reduction
from tensorflow.python.framework import combinations

from tensorflow_similarity import losses


# TODO(ovallis): Refactor XBM loss to work with graph mode and update tests.
@combinations.generate(
    combinations.combine(
        mode=["eager"],
    )
)
class XBMLossTest(tf.test.TestCase):
    def test_config(self):
        xbm_tpl_obj = losses.XBM(
            loss=losses.TripletLoss(distance="cosine"),
            memory_size=12,
            reduction=Reduction.SUM,
            name="xbm_triplet_loss",
        )
        self.assertEqual(xbm_tpl_obj.distance.name, "cosine")
        self.assertEqual(xbm_tpl_obj.name, "xbm_triplet_loss")
        self.assertEqual(xbm_tpl_obj.memory_size, 12)
        self.assertEqual(xbm_tpl_obj.reduction, Reduction.SUM)

    def test_xbm_loss(self):
        batch_size = 6
        embed_dim = 16

        embeddings1 = tf.random.uniform(shape=[batch_size, embed_dim])
        labels1 = tf.constant(
            [[1], [1], [2], [2], [3], [3]],
            dtype=tf.int32,
        )

        embeddings2 = tf.random.uniform(shape=[batch_size, embed_dim])
        labels2 = tf.constant(
            [[4], [4], [5], [5], [6], [6]],
            dtype=tf.int32,
        )

        distance = "cosine"
        loss = losses.MultiSimilarityLoss(distance=distance)
        loss_nowarm = losses.XBM(loss, memory_size=12, warmup_steps=0)

        # test enqueue
        loss_nowarm(labels1, embeddings1)
        self.assertAllEqual(loss_nowarm._y_pred_memory.numpy().shape, (batch_size, embed_dim))
        self.assertAllEqual(loss_nowarm._y_true_memory, labels1)

        loss_nowarm(labels2, embeddings2)
        self.assertAllEqual(loss_nowarm._y_pred_memory.numpy().shape, (2 * batch_size, embed_dim))
        self.assertAllEqual(loss_nowarm._y_true_memory, tf.concat([labels2, labels1], axis=0))

        # test dequeue
        loss_nowarm(labels2, embeddings2)
        self.assertAllEqual(loss_nowarm._y_pred_memory.numpy().shape, (2 * batch_size, embed_dim))
        self.assertAllEqual(loss_nowarm._y_true_memory, tf.concat([labels2, labels2], axis=0))

        # test warmup
        loss_warm = losses.XBM(loss, memory_size=12, warmup_steps=1)

        loss_warm(labels1, embeddings1)
        self.assertAllEqual(loss_warm._y_pred_memory.numpy().shape, (0, embed_dim))
        self.assertAllEqual(loss_warm._y_true_memory, tf.constant([[]], shape=(0, 1), dtype=tf.int32))

        loss_warm(labels2, embeddings2)
        self.assertAllEqual(loss_warm._y_pred_memory.numpy().shape, (batch_size, embed_dim))
        self.assertAllEqual(loss_warm._y_true_memory, labels2)
