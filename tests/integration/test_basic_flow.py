import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.models import load_model

from tensorflow_similarity import search, stores
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.samplers import TFDataSampler
from tensorflow_similarity.training_metrics import dist_gap, max_pos, min_neg


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
        vect = [0] * num_classes * reps
        for rep in range(reps):
            idx = i + num_classes * rep
            vect[idx] = 1
        x.extend([vect for _ in range(num_examples_per_class)])
    return tf.constant(x, dtype=tf.keras.backend.floatx()), tf.constant(y, dtype="int32")


@parameterized.named_parameters(
    {"testcase_name": "float32", "precision": "float32", "search_type": "faiss", "store_type": "cached"},
    {"testcase_name": "float16", "precision": "float16", "search_type": "linear", "store_type": "default"},
    {"testcase_name": "mixed_float16", "precision": "mixed_float16", "search_type": "default", "store_type": "memory"},
)
class BasicFlowTest(tf.test.TestCase, parameterized.TestCase):
    def test_basic_flow(self, precision, search_type, store_type):
        tmp_dir = self.get_temp_dir()
        policy = tf.keras.mixed_precision.Policy(precision)
        tf.keras.mixed_precision.set_global_policy(policy)

        NUM_CLASSES = 8
        REPS = 4
        EXAMPLES_PER_CLASS = 64
        CLASS_PER_BATCH = 8
        STEPS_PER_EPOCH = 500
        K = 5
        NUM_MATCHES = 3

        distance = "cosine"
        positive_mining_strategy = "hard"
        negative_mining_strategy = "semi-hard"

        x, y = generate_dataset(NUM_CLASSES, EXAMPLES_PER_CLASS)
        sampler = TFDataSampler(tf.data.Dataset.from_tensor_slices((x, y)), classes_per_batch=CLASS_PER_BATCH)

        # Search
        search_obj = None
        if search_type == "linear":
            search_obj = search.LinearSearch(distance="cosine", dim=4)
        elif search_type == "faiss":
            search_obj = search.FaissSearch(distance="cosine", dim=4, m=4, nlist=8, nprobe=8)

        # Store
        kv_store = None
        if store_type == "cached":
            kv_store = stores.CachedStore(path=tmp_dir)
        if store_type == "memory":
            kv_store = stores.MemoryStore()

        # model
        inputs = tf.keras.layers.Input(shape=(NUM_CLASSES * REPS,))
        # dont use x as variable
        m = tf.keras.layers.Dense(8, activation="relu")(inputs)
        m = tf.keras.layers.Dense(4, activation="relu")(m)
        outputs = MetricEmbedding(4)(m)
        model = SimilarityModel(inputs, outputs)

        # loss
        triplet_loss = TripletLoss(
            distance=distance,
            positive_mining_strategy=positive_mining_strategy,
            negative_mining_strategy=negative_mining_strategy,
        )

        # compile
        metrics = [dist_gap(distance), min_neg(distance), max_pos(distance)]
        compile_params = {"optimizer": "adam", "metrics": metrics, "loss": triplet_loss}
        if search_obj is not None:
            compile_params["search"] = search_obj
        if kv_store is not None:
            compile_params["kv_store"] = kv_store
        model.compile(**compile_params)

        # train
        history = model.fit(sampler, steps_per_epoch=STEPS_PER_EPOCH, epochs=15)

        # check that history is properly filled
        self.assertIn("loss", history.history)
        self.assertIn("dist_gap", history.history)

        # indexing ensuring that index is working
        model.reset_index()
        model.index(x, y)
        self.assertLen(x, model.index_size())

        # # lookup
        neighbors = model.single_lookup(x[0], k=K)
        self.assertLen(neighbors, K)

        # FIXME(ovallis): This seems to produce flakey tests at the moment.
        # check the model returns reasonable matching
        # assert neighbors[0].label == 0

        # check also the last x example which should be for the last class
        neighbors = model.single_lookup(x[-1], k=K)
        self.assertLen(neighbors, K)

        # FIXME(ovallis): This seems to produce flakey tests at the moment.
        # # check the model returns reasonable matching
        # assert neighbors[0].label == NUM_CLASSES - 1

        # batch lookup
        batch_neighbors = model.lookup(x[:10], k=K)
        self.assertLen(batch_neighbors, 10)

        # calibration
        calibration = model.calibrate(x, y, verbose=0)
        # calibration is a DataClass with two attributes.
        self.assertIn("thresholds", calibration.__dict__)
        self.assertIn("cutpoints", calibration.__dict__)

        # # evaluation
        metrics = model.evaluate_classification(x, y)
        self.assertIn("optimal", metrics)
        self.assertGreaterEqual(metrics["optimal"]["precision"], 0)
        self.assertLessEqual(metrics["optimal"]["precision"], 1)
        self.assertGreaterEqual(metrics["optimal"]["recall"], 0)
        self.assertLessEqual(metrics["optimal"]["recall"], 1)

        # matchings
        matches = model.match(x[:NUM_MATCHES], cutpoint="optimal")
        self.assertLen(matches, NUM_MATCHES)

        # # index summary
        model.index_summary()

        # # model save
        model.save(tmp_dir)

        # # model load
        mdl2 = load_model(tmp_dir)
        mdl2.load_index(tmp_dir)
        self.assertEqual(model.index_size(), mdl2.index_size())


if __name__ == "__main__":
    tf.test.main()
