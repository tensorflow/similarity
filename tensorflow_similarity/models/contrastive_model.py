import tensorflow as tf
from termcolor import cprint
from pathlib import Path
import json
from typing import Any, Callable, Dict, Mapping, Optional, Union

# @tf.keras.utils.register_keras_serializable(package="Similarity")


class ContrastiveModel(tf.keras.Model):
    def __init__(
        self, encoder_model, projector_model, swap_representation=False
    ) -> None:
        super().__init__()

        self.encoder = encoder_model
        self.projector = projector_model
        self.swap_representation = swap_representation
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def train_step(self, data):
        if len(data) == 2:
            view1 = data[0]
            view2 = data[1]
        else:
            view1 = data[0]
            view2 = data[0]

        # Forward pass through the encoder and predictor
        with tf.GradientTape() as tape:

            # compute representation
            z1 = self.encoder(view1)
            z2 = self.encoder(view2)

            # compute projection
            p1 = self.projector(z1)
            p2 = self.projector(z2)

            # Allows to swap projections (ala SimSiam)
            if self.swap_representation:
                # SimSiam
                l1_args = (z1, p2)
                l2_args = (z2, p1)
            else:
                # SimCLR
                l1_args = (z1, z2)
                l2_args = (z2, z1)

            l1 = self.compiled_loss(*l1_args)
            l2 = self.compiled_loss(*l2_args)
            loss = tf.math.reduce_mean(l1) + tf.math.reduce_mean(l2)

        # collect train variables from both the encoder and the projector
        tvars = (
            self.encoder.trainable_variables
            + self.projector.trainable_variables
        )

        # Compute gradients
        gradients = tape.gradient(loss, tvars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, tvars))

        # Update metrics
        # !This are contrastive metrics with different input
        # TODO: figure out interesting metrics -- z Mae?
        # TODO: check metrics are of the right type in compile?
        self.compiled_metrics.update_state([z1, z2], [p1, p2])

        # report loss manually
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    # fix TF 2.x < 2.7 bugs when using generator
    def call(self, inputs):
        return inputs

    def get_encoder(self):
        "Return encoder model"
        return self.encoder

    def summary(self):
        cprint("[Encoder]", "green")
        self.encoder.summary()
        cprint("\n[Projector]", "magenta")
        self.projector.summary()

    def save(
        self,
        filepath: Union[str, Path],
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: Optional[str] = None,
        signatures: Optional[Union[Callable, Mapping[str, Callable]]] = None,
        options: Optional[tf.saved_model.SaveOptions] = None,
        save_traces: bool = True,
    ) -> None:
        """Save Constrative model encoder and projector"""
        spath = Path(filepath)
        epath = spath / "encoder"
        ppath = spath / "projector"
        cpath = spath / "config.json"

        cprint("[Saving projector model]", "blue")
        cprint("|-path:%s" % ppath, "green")
        self.projector.save(
            str(ppath),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        cprint("[Saving encoder model]", "blue")
        cprint("|-path:%s" % epath, "green")
        self.encoder.save(
            str(epath),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        with open(str(cpath), "w+") as o:
            config = self.to_config()
            json.dump(config, o)

    def to_config(self) -> Dict[str, Any]:
        return {"swap_representation": self.swap_representation}

    @staticmethod
    def load(path: str):
        raise NotImplementedError()
