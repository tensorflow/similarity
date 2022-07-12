import tensorflow as tf


def convert_sync_batchnorm(model):
    """Replace BatchNormalization layers to SyncBatchNormalization in place.
    WARNINGS:
    * This function is tested only with efficientnet and resnet
    * The returned model has shared layers with the input one. One of them should be disposed.
    """

    layer2newtensor = {}
    in_ = tf.keras.layers.Input(model.input.shape[1:])
    layer2newtensor[model.input.name] = in_
    for layer in model.layers[1:]:
        assert len(layer.inbound_nodes) == 1
        if isinstance(layer.inbound_nodes[0].inbound_layers, list):  # mutliple inputs
            x = [layer2newtensor[in_l.name] for in_l in layer.inbound_nodes[0].inbound_layers]
        else:
            x = layer2newtensor[layer.inbound_nodes[0].inbound_layers.name]
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer = tf.keras.layers.experimental.SyncBatchNormalization(**layer.get_config())

        if "truediv" in layer.name:
            # efficeientnet edge case https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/efficientnet.py#L334
            x = layer(x, layer.inbound_nodes[0]._flat_arguments[1])
        else:
            x = layer(x)

        layer2newtensor[layer.name] = x
    out_ = layer2newtensor[model.layers[-1].name]
    new_model = tf.keras.Model(inputs=in_, outputs=out_, name=model.name)
    new_model.set_weights(model.get_weights())
    return new_model
