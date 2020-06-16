# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class Explainer(object):
    """Explainer class that computes grad cam, grad cam ++ for a given image
       and a given model/tf.similarity model. This class helps users to visualize
       what are the models (specifically last conv layer of the model) is looking
       for for an image.
    """

    def __init__(
            self,
            model,
            layer_name=None,
            algorithm="grad_cam",
            dtype=tf.keras.layers.Conv2D):

        if layer_name is None:
            last_conv_layer = self._get_layers(model, dtype)[-1]
            layer_name = last_conv_layer.name

        # create a grad model for computing the gradient to the conv layer
        # selected
        layer_output = model.get_layer(layer_name).output
        grad_model_input = [model.input]
        grad_model_output = [layer_output, model.output]

        self.grad_model = tf.keras.Model(grad_model_input, grad_model_output)

        self.model = model
        self.is_classification = self._is_classification_model(model)
        self.layer_name = layer_name
        self.algorithm = algorithm
        self.dtype = dtype

    def _is_classification_model(self, model):
        last_layer = model.layers[-1]
        if hasattr(last_layer, "activation"):
            activation = last_layer.activation
            return activation == tf.keras.activations.softmax
        return False

    def _get_layers(self, model, dtype=tf.keras.layers.Conv2D):
        """A small helper method to get layers with given dtype from the model."""
        layers = []
        for layer in model.layers:
            if isinstance(layer, dtype):
                layers.append(layer)
        return layers

    def grad_cam(self, output, grad):
        weights = tf.reduce_mean(grad, axis=(0, 1))
        # Perform ponderated sum : w_i * output[:, :, i]
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        return cam

    def grad_cam_plus(self, output, grad):
        epsilon = 10 ** -10
        first_grad = grad
        second_grad = grad ** 2
        third_grad = grad ** 3

        # compute alphas
        global_sum = tf.reduce_sum(output, axis=[0, 1], keepdims=True)
        alpha_num = second_grad
        alpha_denom = alpha_num * 2 + third_grad * global_sum + epsilon
        alphas = alpha_num / alpha_denom

        # compute weighted alphas
        weights = tf.maximum(first_grad, 0)
        alpha_normalization_constant = tf.reduce_sum(
            alphas, axis=[0, 1], keepdims=True) + epsilon
        alphas /= alpha_normalization_constant
        weighted_alphas = weights * alphas

        deep_linearization_weights = tf.reduce_sum(
            weighted_alphas, axis=[0, 1])

        cam = tf.reduce_sum(
            tf.multiply(
                deep_linearization_weights,
                output),
            axis=-1)
        return cam

    def score_cam(self, output, grad):
        pass

    def explain(self, images, class_ids=None):

        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            conv_outputs, model_loss = self.grad_model(inputs)

            # For classification (softmax) models only, this block selects the
            # loss corresponding to the selected classes or predicted classes
            if class_ids:
                # if given an int as class_ids turn it into an array
                if isinstance(class_ids, int):
                    num_images = len(images)
                    class_ids = np.full(num_images, fill_value=class_ids)
                class_ids = tf.constant(class_ids)
                model_loss = tf.gather(model_loss, class_ids, axis=1)
            elif self.is_classification:
                predicted_classes = tf.math.argmax(model_loss, axis=1)
                model_loss = tf.gather(model_loss, predicted_classes, axis=1)

        grads = tape.gradient(model_loss, conv_outputs)

        conv_mask = tf.cast(conv_outputs > 0, tf.float32)
        grads_mask = tf.cast(grads > 0, tf.float32)
        guided_grads = conv_mask * grads_mask * grads

        # compute heat maps
        heat_maps = []
        for output, grad in zip(conv_outputs, guided_grads):
            if self.algorithm == "grad_cam":
                cam = self.grad_cam(output, grad)
            elif self.algorithm == "grad_cam_plus":
                cam = self.grad_cam_plus(output, grad)

            heat_maps.append(cam)
        heat_maps = np.asarray(heat_maps)

        # resize map so fit output
        original_size = [images.shape[1], images.shape[2]]
        heat_maps = heat_maps[..., np.newaxis]
        heat_maps = tf.image.resize(
            heat_maps,
            original_size,
            "bicubic",
            antialias=True)
        # normalize heat_maps
        mins = tf.reduce_min(heat_maps, axis=[1, 2, 3], keepdims=True)
        maxs = tf.reduce_max(heat_maps, axis=[1, 2, 3], keepdims=True)
        heat_maps = (heat_maps - mins) / (maxs - mins)
        return heat_maps

    def render(
            self,
            original_images,
            heat_maps=None,
            alpha=0.8,
            cmap="viridis"):
        """Render the gradient on the original images, this method overlaps
           original_images and maps.

           Args:
               original_images (np.array): A numpy array of shape
                  (num_images, width, height, num_dims), num_dims should be either
                  1 (gray scale), or 3 (RGB). Since Matlibplot does not take (W, H, 1)
                  as input for images we will remove the last dimension if it is 1.
               heat_maps (np.array): Have the same shape as original_images, this is
                  the heatmaps returned from the explain method.
               alpha (int): How much we want to overlay the heatmap.
               cmap (string): The cmap we want to use.

          Returns:
              figure (matplotlib.Figure): The figure that shows the overlaps of
                  heatmaps and original_images.

        """

        # compute heat maps if users did not supply it
        if heat_maps is None:
            heat_maps = self.explain(original_images)

        # squeeze last dimension if it's 1
        if original_images.shape[-1] == 1:
            original_images = np.squeeze(original_images, axis=-1)
            heat_maps = np.squeeze(heat_maps, axis=-1)

        num_images = original_images.shape[0]
        figure, axes = plt.subplots(nrows=1, ncols=num_images)

        for i, (original_image, heat_map) in enumerate(
                zip(original_images, heat_maps)):
            # the subplot to plot
            if len(original_images) > 1:
                ax = axes[i]
            else:
                ax = axes

            # plot the original image and then overlay it with heat map
            ax.imshow(original_image)
            ax.imshow(heat_map, alpha=alpha, cmap=cmap)

            # removing ticks and grid
            ax.set_xticks([])
            ax.set_yticks([])

        return figure
