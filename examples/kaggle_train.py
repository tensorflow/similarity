"Kaggle commandline training script"
from multiprocess import cpu_count
from collections import defaultdict
import random
import csv
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
from pathlib import Path
from tabulate import tabulate
from time import time

tf.get_logger().setLevel('ERROR')  # silence TF warning
from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.samplers import RandAugment
from tensorflow_similarity.architectures import EfficientNetSim
from tensorflow_similarity.losses import MultiSimilarityLoss
from tensorflow_similarity.visualization import confusion_matrix  # matching performance

EPOCHS = 5000
EMBEDDING_SIZE = 512
IMG_SIZE = 320
STEPS_PER_EPOCHS = 200

CLASSES_PER_BATCH = 128  # batch size = CLASSES_PER_BATCH* 2
NUM_IMGS = 1580470
NUM_LANDMARKS = 203092
NUM_VAL_LANDMAKRS = 300

val_lmk_idxs = random.sample([i for i in range(NUM_LANDMARKS)],
                             k=NUM_VAL_LANDMAKRS)
CPU_COUNT = cpu_count()
print("NUM CPU:", CPU_COUNT)

data_path = 'data/'
x_train = []
y_train = []
x_test_p = []
y_test = []
testlmk2id = {}
f = open(data_path + 'train.csv')
csvreader = csv.reader(f, )
pb = tqdm(total=1580469, desc="loading images path")
for row_id, row in enumerate(csvreader):

    # skip headers
    if row_id > 1:

        # compute path to img
        slug = "/".join([row[0][i] for i in range(3)])
        img_path = data_path + 'train/' + slug + '/' + row[0] + ".jpg"

        # add img either in validation or testing
        cidx = int(row[1])
        if cidx in val_lmk_idxs:
            y_test.append(cidx)
            x_test_p.append(img_path)
            testlmk2id[cidx] = len(testlmk2id)
        else:
            y_train.append(cidx)
            x_train.append(img_path)
        pb.update()
pb.close()

print("train size", len(x_train))
print("test size", len(x_test_p))


def load_img(p):
    "Load image from disk and resize it"
    img = tf.io.read_file(p)
    img = tf.io.decode_image(img)
    return tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)


x_test = []
for p in tqdm(x_test_p):
    img = load_img(p)
    x_test.append(img)

IMG_PER_CLASS = 1
index = defaultdict(list)
x_index = []
y_index = []
x_query = []
y_query = []
for idx in range(len(x_test)):
    cidx = testlmk2id[y_test[idx]]
    if len(index[cidx]) < IMG_PER_CLASS:
        index[cidx].append(x_test[idx])
        x_index.append(x_test[idx])
        y_index.append(cidx)
    else:
        x_query.append(x_test[idx])
        y_query.append(cidx)

#FIXME ADDED CASTINGB
x_test = np.array(x_test)
y_test = np.array(y_test)
x_index = np.array(x_index)
y_index = np.array(y_index)
x_query = np.array(x_query)
y_query = np.array(y_query)


# read each image from disk and construct the batch as "augment function" to
# sampler
@tf.function()
def process(p):
    augmenter = RandAugment()
    img = tf.io.decode_image(p)
    if tf.shape(img)[0] < IMG_SIZE or tf.shape(img)[1] < IMG_SIZE:
        img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    img = tf.image.random_crop(img, (IMG_SIZE, IMG_SIZE, 3))
    img = augmenter.distort(img)
    return img


def loader(x, y, sample_per_batch, is_warmup):

    imgs = []
    for p in x:
        imgs.append(tf.io.read_file(p))
    imgs = tf.stack(imgs)
    imgs = tf.map_fn(process,
                     imgs,
                     parallel_iterations=CPU_COUNT,
                     dtype='uint8')
    # return tf.stack(imgs, axis=0), tf.constant(y)
    return imgs, y


print("Build train sampler")
train_ds = MultiShotMemorySampler(
    x_train,
    y_train,
    #FIXME ADDED step per epichs
    steps_per_epoch=STEPS_PER_EPOCHS,
    classes_per_batch=CLASSES_PER_BATCH,
    augmenter=loader)

x_batch, y_batch = train_ds.generate_batch(1)
model = EfficientNetSim((IMG_SIZE, IMG_SIZE, 3),
                        embedding_size=EMBEDDING_SIZE,
                        augmentation=None)
model.summary()

slug = "%s_%s_%s_%s" % (EMBEDDING_SIZE, IMG_SIZE, CLASSES_PER_BATCH, time())
cb = [
    tf.keras.callbacks.TensorBoard(log_dir='logs/%s' % slug,
                                   update_freq='batch'),
    tf.keras.callbacks.ModelCheckpoint('models/%s' % slug)
]
loss = MultiSimilarityLoss(distance='cosine')
model.compile('adam', loss=loss)
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=cb)
