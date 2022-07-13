from unittest import mock

import tensorflow as tf

from tensorflow_similarity.visualization import confusion_matrix


def test_confusion_matrix_normalize_true():
    y_true = tf.constant([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = tf.constant([0, 1, 2, 4, 3, 0, 1, 4, 4, 4])

    _, cm = confusion_matrix(y_pred, y_true, show=False)

    expected_cm = tf.constant(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ],
        dtype="float32",
    )

    assert tf.reduce_all(tf.math.equal(cm, expected_cm))


def test_confusion_matrix_normalize_false():
    y_true = tf.constant([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = tf.constant([0, 1, 2, 4, 3, 0, 1, 4, 4, 4])

    _, cm = confusion_matrix(y_pred, y_true, normalize=False, show=False)

    expected_cm = tf.constant(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype="float32",
    )

    assert tf.reduce_all(tf.math.equal(cm, expected_cm))


def test_confusion_matrix_plot():
    y_true = tf.constant([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = tf.constant([0, 1, 2, 4, 3, 0, 1, 4, 4, 4])

    ax, _ = confusion_matrix(y_pred, y_true, normalize=True, show=False)

    expected_cell_text = [
        "1.00",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1.00",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0.50",
        "0",
        "0.50",
        "0",
        "0",
        "0",
        "0",
        "1.00",
        "0",
        "0",
        "0",
        "0.50",
        "0.50",
    ]
    tl = ["−1", "0", "1", "2", "3", "4", "5"]

    # Check the cell text
    assert all([a._text == b for a, b in zip(ax.texts, expected_cell_text)])
    # Check the axes text
    assert ax.title._text == "Confusion matrix"
    assert ax.get_xlabel() == "Predicted label\naccuracy=0.6000; misclass=0.4000"
    assert ax.get_ylabel() == "True label"
    # Check the ticks are just integers.
    assert all([a._text == b for a, b in zip(ax.get_xticklabels(), tl)])
    assert all([a._text == b for a, b in zip(ax.get_yticklabels(), tl)])


def test_confusion_matrix_plot_labels():
    y_true = tf.constant([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = tf.constant([0, 1, 2, 4, 3, 0, 1, 4, 4, 4])
    lbls = ["A", "B", "C", "D", "E"]

    ax, _ = confusion_matrix(y_pred, y_true, labels=lbls, show=False)

    # Check all tick labels match the provided list
    assert all([a._text == b for a, b in zip(ax.get_xticklabels(), lbls)])
    assert all([a._text == b for a, b in zip(ax.get_yticklabels(), lbls)])


def test_confusion_matrix_title():
    y_true = tf.constant([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = tf.constant([0, 1, 2, 4, 3, 0, 1, 4, 4, 4])

    ax, _ = confusion_matrix(y_pred, y_true, title="foo", show=False)

    # Check the main title
    assert ax.title._text == "foo"


@mock.patch("tensorflow_similarity.visualization.confusion_matrix_viz.plt.show")
@mock.patch(
    "tensorflow_similarity.visualization.confusion_matrix_viz.plt.subplots",
    autospec=True,
)
def test_confusion_matrix_mock(mock_subplots, mock_show):
    ax = mock.Mock()
    fig = mock.Mock()
    mock_subplots.return_value = (fig, ax)

    y_true = tf.constant([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = tf.constant([0, 1, 2, 4, 3, 0, 1, 4, 4, 4])

    _, cm = confusion_matrix(y_pred, y_true, cmap="Reds", show=False)
    mock_subplots.assert_called_with(figsize=(8, 6))
    ax.imshow.assert_called_with(cm, interpolation="nearest", cmap="Reds")
    fig.colorbar.assert_called_with(ax.imshow())
    mock_show.assert_not_called()

    _, cm = confusion_matrix(y_pred, y_true, cmap="Reds", show=True)
    mock_show.assert_called_once()
