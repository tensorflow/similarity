import tensorflow as tf
from tensorflow.keras.models import Model


class SimilarityModel(Model):
    """Sub-classing Keras.Model to allow access to the forward pass values for
    efficient metric-learning.
    """

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y,
                                      y_pred,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # FIXME: add our custom metrics and storage of vector here

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
