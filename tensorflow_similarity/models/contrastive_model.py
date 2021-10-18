import tensorflow as tf
from termcolor import cprint

# @tf.keras.utils.register_keras_serializable(package="Similarity")
class ContrastiveModel(tf.keras.Model):

    def __init__(self,
                 encoder_model,
                 projector_model,
                 swap_representation=False) -> None:
        super(ContrastiveModel, self).__init__()

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

            # Allows to swap projections (ala SiamSiam)
            if self.swap_representation:
                l1_args = (z1, p2)
                l2_args = (z2, p1)
            else:
                l1_args = (z1, p1)
                l2_args = (z2, p2)

            l1 = self.compiled_loss(*l1_args)
            l2 = self.compiled_loss(*l2_args)
            loss = (l1 + l2)

        # collect train variables from both the encoder and the projector
        tvars = self.encoder.trainable_variables + self.projector.trainable_variables

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
        cprint("[Encoder]", 'green')
        self.encoder.summary()
        cprint("\n[Projector]", 'magenta')
        self.projector.summary()

    def save(self):
        raise NotImplementedError()
