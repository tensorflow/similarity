"Kaggle commandline training script"
import argparse
from collections import defaultdict
import csv
from multiprocessing import cpu_count
import os
from pathlib import Path
from typing import Tuple
import random

from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow import Tensor
import numpy as np
from time import time

from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.samplers import RandAugment
from tensorflow_similarity.architectures import EfficientNetSim
from tensorflow_similarity.losses import MultiSimilarityLoss

tf.get_logger().setLevel("ERROR")  # silence TF warning

parser = argparse.ArgumentParser(
    description="Kaggle script for training via command line.")
parser.add_argument('--data_path',
                    default='data',
                    help='Path to load the dataset and write the results.')
args = parser.parse_args()

DATA_PATH = Path(args.data_path)
EPOCHS = 5000
EMBEDDING_SIZE = 512
IMG_SIZE = 320
STEPS_PER_EPOCHS = 200

CLASSES_PER_BATCH = 128  # batch size = CLASSES_PER_BATCH*2
NUM_IMGS = 1580470
NUM_LANDMARKS = 203092
NUM_VAL_LANDMAKRS = 300
CPU_COUNT = cpu_count()
print("NUM CPU:", CPU_COUNT)


def load_img(img_path: str) -> Tensor:
    "Load image from disk and resize it"
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img)
    return tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)


# read each image from disk and construct the batch as "augment function" to
# sampler
@tf.function()  # noqa
def process(img: Tensor) -> Tensor:
    augmenter = RandAugment()
    img = tf.io.decode_image(img)
    if tf.shape(img)[0] < IMG_SIZE or tf.shape(img)[1] < IMG_SIZE:
        img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    img = tf.image.random_crop(img, (IMG_SIZE, IMG_SIZE, 3))
    img = augmenter.distort(img)
    return img


def loader(x: np.ndarray, y: np.ndarray, sample_per_batch: int,
           is_warmup: bool) -> Tuple[Tensor, np.ndarray]:
    imgs = []
    for img_path in x:
        imgs.append(tf.io.read_file(img_path))
    imgs = tf.stack(imgs)
    imgs = tf.map_fn(process,
                     imgs,
                     parallel_iterations=CPU_COUNT,
                     dtype="uint8")

    return imgs, y


val_lmk_idxs = random.sample([i for i in range(NUM_LANDMARKS)],
                             k=NUM_VAL_LANDMAKRS)
x_train_img_path = []
y_train = []
x_test_img_path = []
y_test = []
testlmk2id = {}

pb = tqdm(total=NUM_IMGS, desc="loading images path")

with open(DATA_PATH / "train.csv") as csvfile:
    # pathlib is slow while looping
    data_path_str = str(DATA_PATH)

    # reader is faster than DictReader
    reader = csv.reader(csvfile)
    next(reader)  # drop the header

    for row in reader:
        # compute path to img
        slug = [row[0][i] for i in range(3)]
        img_path = os.path.join(data_path_str, "train", *slug, f"{row[0]}.jpg")

        # add img either in validation or testing
        class_idx = int(row[1])
        if class_idx in val_lmk_idxs:
            y_test.append(class_idx)
            x_test_img_path.append(img_path)
            testlmk2id[class_idx] = len(testlmk2id)
        else:
            y_train.append(class_idx)
            x_train_img_path.append(img_path)
        pb.update()
pb.close()

print("train size", len(x_train_img_path))
print("test size", len(x_test_img_path))

x_test = []
for img_path in tqdm(x_test_img_path):
    img = load_img(img_path)
    x_test.append(img)

IMG_PER_CLASS = 1
index = defaultdict(list)
x_index = []
y_index = []
x_query = []
y_query = []

for idx in range(len(x_test)):
    class_idx = testlmk2id[y_test[idx]]
    img = x_test[idx]
    if len(index[class_idx]) < IMG_PER_CLASS:
        index[class_idx].append(img)
        x_index.append(img)
        y_index.append(class_idx)
    else:
        x_query.append(img)
        y_query.append(class_idx)

x_test = np.array(x_test)
y_test = np.array(y_test)
x_index = np.array(x_index)
y_index = np.array(y_index)
x_query = np.array(x_query)
y_query = np.array(y_query)

print("Build train sampler")
train_ds = MultiShotMemorySampler(
    x_train_img_path,
    y_train,
    # FIXME ADDED step per epochs
    steps_per_epoch=STEPS_PER_EPOCHS,
    classes_per_batch=CLASSES_PER_BATCH,
    augmenter=loader,
)

model = EfficientNetSim((IMG_SIZE, IMG_SIZE, 3),
                        embedding_size=EMBEDDING_SIZE,
                        augmentation=None)
model.summary()

slug = f"{EMBEDDING_SIZE}_{IMG_SIZE}_{CLASSES_PER_BATCH}_{time()}"
cb = [
    tf.keras.callbacks.TensorBoard(log_dir=DATA_PATH / "logs" / slug,
                                   update_freq="batch"),
    tf.keras.callbacks.ModelCheckpoint(DATA_PATH / "models" / slug),
]
loss = MultiSimilarityLoss(distance="cosine")
model.compile("adam", loss=loss)
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=cb)
