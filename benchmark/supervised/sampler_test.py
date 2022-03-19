#%%
import tensorflow_similarity
tensorflow_similarity.utils.tf_cap_memory()

from tensorflow_similarity.samplers import TFRecordDatasetSampler, TFDatasetMultiShotMemorySampler
import os
from benchmark import _parse_image_function
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
full_path = "../../datasets/1.0.0/cars196/index"
full_path = os.path.join(os.path.dirname(__file__), full_path)
sampler = TFRecordDatasetSampler(
    shard_path=full_path,
    deserialization_fn=_parse_image_function,
    example_per_class=196 // 2,
    batch_size=128,
    shard_suffix="*.tfrecords",
    examples_per_class_per_batch=4
)


# %%
for item in sampler.take(1):
    plt.imshow(item[0][0])
# %%
count = 0
for i in sampler:
    count += 1
    print(count)

print(count)
# %%


with tf.device("/cpu:0"):
    def preprocess(x, y):
        size = 360
        def resize(img, size):
            with tf.device("/cpu:0"):
                return tf.image.resize_with_pad(img, size, size)

        return resize(x, size) / 255, y

    examples_per_class_per_batch = 4
    train_cls = list(range(196//2))
    train_ds = TFDatasetMultiShotMemorySampler(
        "cars196",
        splits="train",
        examples_per_class_per_batch=examples_per_class_per_batch,
        classes_per_batch=196 // 2,
        preprocess_fn=preprocess,
        class_list=train_cls,
        # augmenter=img_augmentation,
    )  # We filter train data to only keep the train classes.
# %%
with tf.device("/cpu:0"):
    def preprocess(x, y):
        size = 360
        def resize(img, size):
            with tf.device("/cpu:0"):
                return tf.image.resize_with_pad(img, size, size)

        return resize(x, size) / 255, y

    examples_per_class_per_batch = 4
    test_cls = list(range(196//2,196))
    test_ds = TFDatasetMultiShotMemorySampler(
        "cars196",
        splits="test",
        examples_per_class_per_batch=examples_per_class_per_batch,
        classes_per_batch=196 // 2,
        preprocess_fn=preprocess,
        class_list=train_cls,
        # augmenter=img_augmentation,
    )
# %%
num_targets = 200  # @param {type:"integer"}
num_queries = 300  # @param {type:"integer"}
k = 3  # @param {type:"integer"}
# log_dir = "logs/%d/" % (time())

# Setup EvalCallback by splitting the test data into targets and queries.
queries_x, queries_y = test_ds.get_slice(0, num_queries)
targets_x, targets_y = test_ds.get_slice(num_queries, num_targets)
tsc = tensorflow_similarity.callbacks.EvalCallback(
    queries_x,
    queries_y,
    targets_x,
    targets_y,
    metrics=["f1"],
    k=k,
    # tb_logdir=log_dir  # uncomment if you want to track in tensorboard
)

# Setup SplitValidation callback.
val_loss = tensorflow_similarity.callbacks.SplitValidationLoss(
    queries_x,
    queries_y,
    targets_x,
    targets_y,
    metrics=["binary_accuracy"],
    known_classes=tf.constant(train_cls),
    k=k,
    # tb_logdir=log_dir  # uncomment if you want to track in tensorboard
)

# Adding the Tensorboard callback to track metrics in tensorboard.
# tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # uncomment if you want to track in tensorboard

callbacks = [
    val_loss,
    tsc,
    # tbc # uncomment if you want to track in tensorboard
]

#%%
embedding_size = 128  # @param {type:"integer"}

# building model
model = tensorflow_similarity.architectures.EfficientNetSim(
    train_ds.example_shape, 
    embedding_size,
    pooling="gem",    # Can change to use `gem` -> GeneralizedMeanPooling2D
    gem_p=3.0,        # Increase the contrast between activations in the feature map.
)

epochs = 5  # @param {type:"integer"}
LR = 0.0001  # @param {type:"number"}
gamma = 256  # @param {type:"integer"} # Loss hyper-parameter. 256 works well here.
steps_per_epoch = 100
val_steps = 50


# init similarity loss
loss = tensorflow_similarity.losses.CircleLoss(gamma=gamma)

# compiling and training
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss)
history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=val_steps,
    callbacks=callbacks,
)
# %%
