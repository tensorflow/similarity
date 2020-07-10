from tqdm.auto import tqdm
import tensorflow as tf
from .utils import get_layers

def build_gradient_model(model, layer_name):
    if layer_name is None:
        last_conv_layer = get_layers(model)[-1]
        layer_name = last_conv_layer.name

    # create a grad model for computing the gradient
    # to the conv layer selected
    layer_output = model.get_layer(layer_name).output
    layer_output.activation = 'linear'
    grad_model_input = [model.input]
    grad_model_output = [layer_output, model.output]

    return tf.keras.Model(grad_model_input, grad_model_output)


# @tf.function()
def compute_gradient(model, inputs, class_ids):
    "Compute gradients for a given model and inputs"
    with tf.GradientTape() as tape:
        outputs, predictions = model(inputs)
        model_loss = tf.gather(predictions, class_ids, axis=1)

    grads = tape.gradient(model_loss, outputs)
    return outputs, grads


# @tf.function()
def compute_guided_gradient(outputs, grads):
    conv_mask = tf.cast(outputs > 0, tf.float32)
    grads_mask = tf.cast(grads > 0, tf.float32)
    grads = conv_mask * grads_mask * grads
    return grads


def compute_cam_plus_plus(outputs, gradients):
    maps = []
    for output, grad in zip(outputs, gradients):
        epsilon = 10**-10
        output_exp = tf.math.exp(output)
        first_grad = output_exp * grad
        second_grad = output_exp * grad**2
        third_grad = output_exp * grad**3

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

        deep_linearization_weights = tf.reduce_sum(weighted_alphas,
                                                   axis=[0, 1])

        cam = tf.reduce_sum(tf.multiply(deep_linearization_weights, output),
                            axis=-1)
        maps.append(cam)
    return maps


def grad_cam_plus_plus(grad_model,
                       inputs,
                       class_ids,
                       batch_size,
                       use_guided_gradient=False,
                       verbose=1):
    """
    Apply Grad CAM++ on a model for a given input
    Paper: [Grad-CAM++: Improved Visual Explanations for Deep
    Convolutional Networks](https://arxiv.org/abs/1710.11063)
    """
    total = len(inputs)
    if verbose:
        pb = tqdm(total=total, desc='Explaining')
    maps = []
    for start in range(0, total, batch_size):
        batch = inputs[start:start + batch_size]
        outputs, gradients = compute_gradient(grad_model, batch, class_ids)

        if use_guided_gradient:
            gradients = compute_guided_gradient(outputs, gradients)
        batch_maps = compute_cam_plus_plus(outputs, gradients)
        maps.extend(batch_maps)
        if verbose:
            pb.update(len(batch))
    if verbose:
        pb.close()
    return normalize_heatmap(maps)
  
def normalize_heatmap(maps):
    "Normalize maps between 0 and 1"
    mins = tf.reduce_min(maps, axis=-1, keepdims=True)
    maxs = tf.reduce_max(maps, axis=-1, keepdims=True)
    maps = (maps - mins) / (maxs - mins)
    return maps