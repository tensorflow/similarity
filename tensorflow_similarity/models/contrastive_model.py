from pathlib import Path
import json
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import tensorflow as tf
from tensorflow_similarity.types import FloatTensor
from termcolor import cprint


# @tf.keras.utils.register_keras_serializable(package="Similarity")
class ContrastiveModel(tf.keras.Model):
    def __init__(
        self,
        backbone_model: tf.keras.Model,
        projector_model: tf.keras.Model,
        predictor_model: Optional[tf.keras.Model] = None,
        method: str = "simsiam",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.backbone = backbone_model
        self.projector = projector_model
        self.predictor = predictor_model
        self.method = method
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.supported_methods = ("simsiam", "simclr", "barlow")

        if self.method not in self.supported_methods:
            raise ValueError(
                f"{self.method} is not a supported method."
                f"Supported methods are {self.supported_methods}."
            )

    @tf.function
    def train_step(self, data):
        view1, view2 = self._parse_views(data)

        # Forward pass through the encoder and predictor
        with tf.GradientTape() as tape:
            h1 = self.backbone(view1)
            h2 = self.backbone(view2)

            z1 = self.projector(h1)
            z2 = self.projector(h2)

            if self.predictor:
                p1 = self.predictor(z1)
                p2 = self.predictor(z2)

            if self.method == "simsiam":
                h1 = tf.stop_gradient(h1)
                h2 = tf.stop_gradient(h2)
                l1 = self.compiled_loss(z1, p2)
                l2 = self.compiled_loss(z2, p1)
            elif self.method == "simclr":
                l1 = self.compiled_loss(z1, z2)
                l2 = self.compiled_loss(z2, z1)
            elif self.method == "barlow":
                l1 = self.compiled_loss(z1, z2)
                l2 = 0

            loss = l1 + l2
        # collect train variables from both the encoder and the projector
        tvars = self.backbone.trainable_variables
        tvars += self.projector.trainable_variables
        if self.predictor:
            tvars += self.predictor.trainable_variables

        # Compute gradients
        gradients = tape.gradient(loss, tvars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, tvars))

        # Update metrics
        # !This are contrastive metrics with different input
        # TODO: figure out interesting metrics -- z Mae?
        # TODO: check metrics are of the right type in compile?
        if self.predictor:
            y = [p2, p1]
        else:
            y = [z2, z1]
        self.compiled_metrics.update_state([z1, z2], y)

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
        backbone_path = spath / "backbone"
        proj_path = spath / "projector"
        pred_path = spath / "predictor"
        config_path = spath / "config.json"

        cprint("[Saving backbone model]", "blue")
        cprint("|-path:%s" % backbone_path, "green")
        self.backbone.save(
            str(backbone_path),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        cprint("[Saving projector model]", "blue")
        cprint("|-path:%s" % proj_path, "green")
        self.projector.save(
            str(proj_path),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        cprint("[Saving predictor model]", "blue")
        cprint("|-path:%s" % pred_path, "green")
        self.projector.save(
            str(pred_path),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        super().save(
            str(spath),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        with open(str(config_path), "w+") as o:
            config = self.get_config()
            json.dump(config, o)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "backbone_model": self.backbone_model,
            "projector_model": self.projector_model,
            "predictor_model": self.predictor_model,
            "method": self.method,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _parse_views(
        self, data: Sequence[FloatTensor]
    ) -> Tuple[FloatTensor, FloatTensor]:
        if len(data) == 2:
            view1 = data[0]
            view2 = data[1]
        else:
            view1 = data[0]
            view2 = data[0]

        return view1, view2

    def predict(
        self,
        x: FloatTensor,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        steps: Optional[int] = None,
        callbacks: Optional[tf.keras.callbacks.Callback] = None,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
    ) -> FloatTensor:
        # Here we assume the backbone has a single output layer.
        output: FloatTensor = self.backbone(
            x,
            batch_size,
            verbose,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing,
        )

        return output
