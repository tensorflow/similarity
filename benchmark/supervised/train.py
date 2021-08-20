"Supervised loss benchmark"
import json
import numpy as np
import argparse
from termcolor import cprint
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_similarity.architectures import EfficientNetSim
from tensorflow_similarity.losses import TripletLoss, CircleLoss, PNLoss
from benchmark import load_dataset, clean_dir


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
        x_train, y_train = load_dataset(version, dataset_name, 'train')
        x_test, y_test = load_dataset(version, dataset_name, 'test')
        print("shapes x:", x_train.shape, 'y:', y_train.shape)

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
                                    trainable=trainable)

            model.compile(optimizer=optim, loss=loss)
            history = model.fit(x_train,
                                y_train,
                                batch_size=batch_size,
                                steps_per_epoch=train_steps,
                                epochs=epochs,
                                validation_data=(x_test, y_test),
                                callbacks=callbacks,
                                validation_steps=val_steps)
            # save history
            with open("%shistory.json" % stub, 'w') as o:
                o.write(json.dumps(history.history))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', '-c', help='config path')
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
