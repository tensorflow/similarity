import numpy as np
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt


def viz_neigbors_imgs(example,
                      example_class,
                      neighbors,
                      labels=None,
                      fig_size=(24, 4),
                      cmap='viridis'):
    """Display images nearest neighboors

    Args:
        example ([type]): [description]
        example_class ([type]): [description]
        neighbors ([type]): [description]
        labels ([type], optional): [description]. Defaults to None.
        fig_size ([type], optional): [description]. Defaults to (24, 4).
        cmap ([type], optional): [description]. Defaults to 'viridis'.
    """
    num_cols = len(neighbors) + 1
    plt.subplots(1, num_cols, figsize=fig_size)
    plt_idx = 1

    # draw target
    plt.subplot(1, num_cols, plt_idx)
    plt.imshow(example, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

    val = labels[example_class] if labels else example_class
    plt.title('class %s' % val)
    plt_idx += 1

    for nbg in neighbors:
        plt.subplot(1, num_cols, plt_idx)
        val = labels[nbg.label] if labels else nbg.label
        legend = "%s - d:%.5f" % (val, nbg.distance)
        if nbg.label == example_class:
            color = cmap
        else:
            color = 'Reds'
        plt.imshow(nbg.data, cmap=color)
        plt.title(legend)
        plt.xticks([])
        plt.yticks([])

        plt_idx += 1
    plt.show()


def confusion_matrix(y_pred,
                     y_true,
                     normalize=True,
                     labels=None,
                     title='Confusion matrix',
                     cmap='Blues'):
    """Plot confusion matrix

    Args:
        y_pred: Model prediction returned by `model.match()`

        y_true: Expected class_id.

        normalize: Normalizes matrix values between 0 and 1.
        Defaults to True.

        labels: List of class string label to display instead of the class
        numerical ids. Defaults to None.

        title: Title of the confusion matrix. Defaults to 'Confusion matrix'.

        cmap: Color schema as CMAP. Defaults to 'Blues'.
    """

    cm = np.array(tf.math.confusion_matrix(y_true, y_pred))
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        color = "white" if val > thresh else "black"
        txt = "%.2f" % val if val > 0.0 else "0"
        plt.text(j, i, txt, horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()
