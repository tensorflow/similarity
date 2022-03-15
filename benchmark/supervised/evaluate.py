#%%
from benchmark import load_tfrecord_unbatched_dataset
import tensorflow as tf
from tensorflow_similarity.losses.circle_loss import CircleLoss
from tensorflow_similarity.losses.pn_loss import PNLoss
from tensorflow_similarity.models.similarity_model import SimilarityModel
import os
import numpy as np
from tensorflow_similarity.visualization import viz_neigbors_imgs, confusion_matrix
from tensorflow_similarity.retrieval_metrics.recall_at_k import RecallAtK

#%%
def load_data(version, dataset_name):
    ds_index = load_tfrecord_unbatched_dataset(version, dataset_name, "index")
    ds_seen_queries = load_tfrecord_unbatched_dataset(version, dataset_name, "seen_queries")
    ds_unseen_queries = load_tfrecord_unbatched_dataset(version, dataset_name, "unseen_queries")

    return ds_index, ds_seen_queries, ds_unseen_queries

def load_model(version, dataset_name, model_loss):
    custom_objects = {"Similarity>CircleLoss": CircleLoss, "Similarity>PNLoss": PNLoss}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model =  tf.keras.models.load_model(f'models/{version}/{dataset_name}_{model_loss}')
        sim_model = SimilarityModel.from_config(model.get_config())
        return sim_model

#%%
#UNCOMMENT IF RUNNING IN VSCODE
# os.chdir('../../')
#%%
# def ds_get_cardinality(ds):
#     i = 0
#     for _ in ds:
#         i += 1
#     return ds.apply(tf.data.experimental.assert_cardinality(i))



ds_index, ds_seen_queries, ds_unseen_queries = load_data("1.0.0", "cars196")
# ds_index = ds_get_cardinality(ds_index)
# ds_seen_queries = ds_get_cardinality(ds_seen_queries)
# ds_unseen_queries = ds_get_cardinality(ds_unseen_queries)

ds_index_x, ds_index_y = tf.data.Dataset.get_single_element(ds_index.batch(len(ds_index)))
# ds_index_x, ds_index_y = zip(*list(ds_index.as_numpy_iterator()))
# ds_index_x, ds_index_y = np.array(ds_index_x), np.array(ds_index_y)

ds_seen_queries_x, ds_seen_queries_y = tf.data.Dataset.get_single_element(ds_seen_queries.batch(len(ds_unseen_queries)))
# ds_seen_queries_x, ds_seen_queries_y = zip(*list(ds_seen_queries.as_numpy_iterator()))
# ds_seen_queries_x, ds_seen_queries_y = np.array(ds_seen_queries_x), np.array(ds_seen_queries_y)

ds_unseen_queries_x, ds_unseen_queries_y = tf.data.Dataset.get_single_element(ds_unseen_queries.batch(len(ds_unseen_queries)))
# ds_unseen_queries_x, ds_unseen_queries_y = zip(*list(ds_unseen_queries.as_numpy_iterator()))
# ds_unseen_queries_x, ds_unseen_queries_y = np.array(ds_unseen_queries_x), np.array(ds_unseen_queries_y)
#%%
model = load_model("1.0.0", "cars196", "circle_loss")
# %%
# init similarity loss
loss = CircleLoss(gamma=256)

# compiling and training
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=loss)
# %%
model.reset_index()
model.index(ds_index_x, ds_index_y, data=ds_index_x)
# %%
num_neighbors = 5
nns = model.single_lookup(ds_seen_queries_x[61], k=num_neighbors)
# %%
viz_neigbors_imgs(ds_seen_queries_x[61], ds_seen_queries_y[61], nns)

#%%
calibration = model.calibrate(
    ds_seen_queries_x,
    ds_seen_queries_y,
    calibration_metric="f1",
    matcher="match_nearest",
    extra_metrics=["precision", "recall", "binary_accuracy"],
    verbose=1,
)
# %%

matches = model.match(ds_seen_queries_x, cutpoint="optimal", no_match_label=10)

confusion_matrix(
    matches,
    ds_seen_queries_y,
    # labels=labels,
    title="Confusion matrix for cutpoint:%s" % "optimal",
    normalize=False,
)


# %%
model.evaluate_retrieval(
    ds_seen_queries_x, 
    ds_seen_queries_y, 
    retrieval_metrics=[RecallAtK(k=1), RecallAtK(k=5)]
)

# %%
model.evaluate_retrieval(
    ds_unseen_queries_x, 
    ds_unseen_queries_y, 
    retrieval_metrics=[RecallAtK(k=1), RecallAtK(k=5)]
)
# %%
