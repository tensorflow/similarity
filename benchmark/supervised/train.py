import os

print(os.listdir())

"Supervised loss benchmark"
import json
import numpy as np
import argparse
from termcolor import cprint
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_similarity.architectures import EfficientNetSim, ResNet18Sim
from tensorflow_similarity.losses import (
    TripletLoss,
    CircleLoss,
    PNLoss,
    MultiSimilarityLoss,
)
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.retrieval_metrics import RecallAtK
from tensorflow_similarity.samplers import TFRecordDatasetSampler
import tensorflow as tf
from benchmark import load_dataset, clean_dir, load_tfrecord_dataset, _parse_image_function
import tensorflow_similarity

tensorflow_similarity.utils.tf_cap_memory()


def make_loss(distance, params):
    if params["loss"] == "triplet_loss":
        return TripletLoss(
            distance=distance, negative_mining_strategy=params["negative_mining"]
        )
    elif params["loss"] == "pn_loss":
        return PNLoss(
            distance=distance, negative_mining_strategy=params["negative_mining"]
        )
    elif params["loss"] == "circle_loss":
        return CircleLoss(
            distance=distance, margin=params["margin"], gamma=params["gamma"]
        )
    elif params["loss"] == "multi_similarity":
        return MultiSimilarityLoss(
            distance=distance,
            alpha=params["alpha"],
            beta=params["beta"],
            epsilon=params["epsilon"],
            lmda=params["lambda"],
        )
    else:
        raise ValueError("Unknown loss name", params["loss"])


def test_model(shape, embedding_size):
    inputs = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(embedding_size, activation="relu")(x)
    # outputs = MetricEmbedding(embedding_size)(x)

    model = tf.keras.models.Model(inputs, x)
    return model



def run(config):
    version = config["version"]
    for dataset_name, dconf in config["datasets"].items():
        cprint("[%s]\n" % dataset_name, "yellow")
        batch_size = dconf["batch_size"]
        architecture = dconf["architecture"]
        epochs = dconf["epochs"]
        train_steps = dconf["train_steps"]
        val_steps = dconf["val_steps"]
        shape = dconf["shape"]
        embedding_size = dconf["embedding_size"]
        trainable = dconf["trainable"]
        distance = dconf["distance"]

        cprint("|-loading dataset", "blue")

        USING_TFRECORD = True
        if not USING_TFRECORD:
            x_train, y_train = load_dataset(version, dataset_name, "train")
            x_test, y_test = load_dataset(version, dataset_name, "test")
            print("shapes x:", x_train.shape, "y:", y_train.shape)
        else:
            # # NOTE: Remove repeat if not using train steps
            # train_ds = load_tfrecord_dataset(
            #     version, dataset_name, "train", batch_size
            # )  # .repeat(60)
            # test_ds = load_tfrecord_dataset(
            #     version, dataset_name, "test", batch_size
            # )  # .repeat(60)

            full_path = f"../../datasets/{version}/{dataset_name}/train"
            full_path = os.path.join(os.path.dirname(__file__), full_path)
            train_ds = TFRecordDatasetSampler(
                shard_path=full_path,
                deserialization_fn=_parse_image_function,
                example_per_class=196 // 2,
                batch_size=128,
                shard_suffix="*.tfrecords"
            )

            full_path = f"../../datasets/{version}/{dataset_name}/test"
            full_path = os.path.join(os.path.dirname(__file__), full_path)
            test_ds = TFRecordDatasetSampler(
                shard_path=full_path,
                deserialization_fn=_parse_image_function,
                example_per_class=196 // 2,
                batch_size=128,
                shard_suffix="*.tfrecords"
            )

            # print("Dataset Length", len(train_ds))
            for x, y in train_ds.take(1):
                print("shapes x:", tf.shape(x), "y:", tf.shape(y))

        for lparams in dconf["losses"]:
            cprint("Training %s" % lparams["name"], "green")

            stub = "../../models/%s/%s_%s/" % (version, dataset_name, lparams["name"])
            stub = os.path.join(os.path.dirname(__file__), stub)
            # cleanup dir
            clean_dir(stub)

            # build loss
            loss = make_loss(distance, lparams)
            optim = Adam(lparams["lr"])
            callbacks = [ModelCheckpoint(stub)]

            # TODO: try making your own custom architecture to test if issue has to do with model backbone
            # (i do know that if you don't explicitly declare the trainable attribute in call model won't learn, but should be something to see)
            model = EfficientNetSim(
                shape,
                embedding_size,
                variant=architecture,
                trainable=trainable,
            )

            # model = test_model(shape, embedding_size)

            # model = ResNet18Sim(shape, embedding_size) Try training on colab bc of ram issue

            model.compile(optimizer=optim, loss=loss)

            if not USING_TFRECORD:
                # NOTE: Numpy ds may not work because not enough samples for
                # train steps and batch size. Soved by using .repeat() for tfrecords
                history = model.fit(
                    x_train,
                    y_train,
                    # batch_size=batch_size,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks,
                    validation_steps=val_steps,
                )
            else:
                # NOTE: epochs has been changed to 20
                # NOTE: Batch size has been uncommented as dataset already batched using load_tfrecord_dataset()
                history = model.fit(
                    train_ds,
                    # batch_size=batch_size,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=test_ds,
                    callbacks=callbacks,
                    validation_steps=val_steps,
                )

            # save history
            with open("%shistory.json" % stub, "w") as o:
                o.write(json.dumps(history.history))


if __name__ == "__main__":
    # UNCOMMENT IF RUNNING IN VSCODE
    # os.chdir("./similarity/")
    print(os.listdir())
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", "-c", help="config path")
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
    # should run
