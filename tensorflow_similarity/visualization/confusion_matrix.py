import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_similarity.types import IntTensor


def confusion_matrix(y_pred: IntTensor,
                     y_true: IntTensor,
                     normalize: bool = True,
                     labels: bool = None,
                     title: str = 'Confusion matrix',
                     cmap: str = 'Blues'):
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
