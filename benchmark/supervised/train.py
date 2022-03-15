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
from tensorflow_similarity.losses import TripletLoss, CircleLoss, PNLoss
from tensorflow_similarity.retrieval_metrics import RecallAtK
import tensorflow as tf
from benchmark import load_dataset, clean_dir, load_tfrecord_dataset


def make_loss(distance, params):
    if params['loss'] == "triplet_loss":
        return TripletLoss(distance=distance,
                           negative_mining_strategy=params['negative_mining'])
    elif params['loss'] == "pn_loss":
        return PNLoss(distance=distance,
                      negative_mining_strategy=params['negative_mining'])
    elif params['loss'] == "circle_loss":
        return CircleLoss(distance=distance,
                          margin=params['margin'],
                          gamma=params['gamma'])
    else:
        raise ValueError("Unknown loss name", params['loss'])


def run(config):
    version = config['version']
    for dataset_name, dconf in config['datasets'].items():
        cprint("[%s]\n" % dataset_name, 'yellow')
        batch_size = dconf['batch_size']
        architecture = dconf['architecture']
        epochs = dconf['epochs']
        train_steps = dconf['train_steps']
        val_steps = dconf['val_steps']
        shape = dconf['shape']
        embedding_size = dconf['embedding_size']
        trainable = dconf['trainable']
        distance = dconf['distance']

        cprint("|-loading dataset", 'blue')

        USING_TFRECORD = True
        if not USING_TFRECORD:
            x_train, y_train = load_dataset(version, dataset_name, 'train')
            x_test, y_test = load_dataset(version, dataset_name, 'test')
            print("shapes x:", x_train.shape, 'y:', y_train.shape)
        else:
            #NOTE: Remove repeat
            train_ds = load_tfrecord_dataset(version, dataset_name, 'train', batch_size)#.repeat(60)
            test_ds = load_tfrecord_dataset(version, dataset_name, 'test', batch_size)#.repeat(60)

            print("Dataset Length", len(train_ds))
            for x, y in train_ds.take(1):
                print("shapes x:", tf.shape(x), 'y:', tf.shape(y))

        for lparams in dconf['losses']:
            cprint("Training %s" % lparams['name'], 'green')

            stub = "models/%s/%s_%s/" % (version, dataset_name, lparams['name'])

            # cleanup dir
            clean_dir(stub)

            # build loss
            loss = make_loss(distance, lparams)
            optim = Adam(lparams['lr'])
            callbacks = [ModelCheckpoint(stub)]

            model = EfficientNetSim(shape,
                                    embedding_size,
                                    variant=architecture,
                                    trainable=trainable,
                                )

            # model = ResNet18Sim(shape, embedding_size) Try training on colab bc of ram issue

            model.compile(optimizer=optim, loss=loss)

            if not USING_TFRECORD:
                # NOTE: Numpy ds may not work because not enough samples for 
                # train steps and batch size. Soved by using .repeat() for tfrecords
                history = model.fit(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    steps_per_epoch=train_steps,
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    callbacks=callbacks,
                                    validation_steps=val_steps)
            else:
                #NOTE: epochs has been changed to 20
                history = model.fit(train_ds,
                            # batch_size=batch_size,
                            # steps_per_epoch=train_steps,
                            epochs=epochs,
                            validation_data=test_ds,
                            callbacks=callbacks,
                            validation_steps=val_steps)

            # save history
            with open("%shistory.json" % stub, 'w') as o:
                o.write(json.dumps(history.history))



if __name__ == '__main__':
    #UNCOMMENT IF RUNNING IN VSCODE
    # os.chdir("./similarity/")
    print(os.listdir())
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', '-c', help='config path')
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
    # should run