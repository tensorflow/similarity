import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow_similarity.indexer import Indexer
from .metrics import metric_name_canonializer


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

            # FIXME: callback
            # self.distances = cosine_distance(y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # self.distance_metrics.update_state(distances)

        # FIXME: add our custom metrics and storage of vector here

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def compile(self,
                optimizer='rmsprop',
                distance='auto',
                loss=None,
                metrics=None,
                distance_metrics=None,
                loss_weights=None,
                mapper='memory',
                matcher='hnsw',
                stat_buffer_size=100,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        "Configures the model for training"

        # Fetching the distance used from the first loss if auto
        if distance == 'auto':
            if isinstance(loss, list):
                metric_loss = loss[0]
            else:
                metric_loss = loss

            try:
                self.distance = metric_loss.distance
            except:  # noqa
                raise ValueError("distance='auto' only works if the first loss\
                     is a metric loss"                                      )
        else:
            self.distance = metric_name_canonializer(distance)

        self._index = Indexer(distance=self.distance,
                              mapper=mapper,
                              matcher=matcher,
                              stat_buffer_size=stat_buffer_size)

        # call underlying keras method
        super().compile(optimizer=optimizer,
                        loss=loss,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        **kwargs)

    def index(self, x, y=None, store_data=True, build=True, verbose=1):
        embeddings = self.predict(x)
        data = x if store_data else None
        self._index.batch_add(embeddings, y, data, build=build, verbose=1)

    def index_reset(self):
        "Reinitialize the index"
        self._index.reset()

    def lookup(self, x, k=5, threads=4):
        embeddings = self.predict(x)
        return self._index.batch_lookup(embeddings, k=k)

    def single_lookup(self, x, k=5):
        x = tf.expand_dims(x, axis=0)
        embedding = self.predict(x)[0]
        return self._index.single_lookup(embedding, k=k)

    def index_summary(self):
        self._index.print_stats()

