from unittest import mock
import pytest

import tensorflow as tf

from tensorflow_similarity.visualization import viz_neigbors_imgs
from tensorflow_similarity.visualization.neighbors_viz import _get_class_label
from tensorflow_similarity.types import Lookup


class TestGetClassLabel:

    @pytest.fixture
    def class_mapping(self):
        return {0: "foo", 1: "bar"}

    def test_example_class_is_none(self, class_mapping):
        c_lbl = _get_class_label(None, class_mapping)
        assert c_lbl == "No Label"

    def test_class_mapping_is_none(self):
        c_lbl = _get_class_label(0, None)
        assert c_lbl == "0"

    def test_get_class_label(self, class_mapping):
        c_lbl = _get_class_label(0, class_mapping)
        assert c_lbl == "foo"

        c_lbl = _get_class_label(1, class_mapping)
        assert c_lbl == "bar"

    def test_example_class_not_in_mapping(self, class_mapping):
        c_lbl = _get_class_label(2, class_mapping)
        assert c_lbl == "2"

    def test_class_mapping_must_implement_get(self):
        msg = "'list' object has no attribute 'get'"
        with pytest.raises(AttributeError, match=msg):
            _ = _get_class_label(0, ["foo", "bar"])


@mock.patch("tensorflow_similarity.visualization.neighbors_viz.plt.show")
@mock.patch(
    "tensorflow_similarity.visualization.neighbors_viz.plt.subplots",
    autospec=True,
)
def test_viz_neighbors_imgs(mock_subplots, mock_show):
    ax_0 = mock.Mock()
    ax_1 = mock.Mock()
    ax_2 = mock.Mock()
    ax_3 = mock.Mock()
    mock_subplots.return_value = (None, [ax_0, ax_1, ax_2, ax_3])

    query_img = tf.constant([1.0])
    nn = [
        Lookup(
            rank=0,  # Incorrect class but class id in class mapping
            distance=0.2,
            label=2,
            data=tf.constant([2.0]),
        ),
        Lookup(
            rank=0,  # Incorrect class and class if not in mapping
            distance=0.3,
            label=3,
            data=tf.constant([3.0]),
        ),
        Lookup(
            rank=0,  # Correct class and class in mapping
            distance=0.1,
            label=1,
            data=tf.constant([4.0]),
        ),
    ]

    viz_neigbors_imgs(
        example=query_img,
        example_class=1,
        neighbors=nn,
        class_mapping={
            1: "foo",
            2: "bar"
        },
        fig_size=(10, 10),
        cmap="Blues",
    )

    mock_subplots.assert_called_with(nrows=1, ncols=4, figsize=(10, 10))

    ax_0.imshow.assert_called_with(query_img, cmap="Blues")
    ax_0.set_xticks.assert_called_with([])
    ax_0.set_yticks.assert_called_with([])
    ax_0.set_title.assert_called_with("foo")

    ax_1.imshow.assert_called_with(tf.constant([2.0]), cmap="Reds")
    ax_1.set_xticks.assert_called_with([])
    ax_1.set_yticks.assert_called_with([])
    ax_1.set_title.assert_called_with("bar - 0.20000")

    ax_2.imshow.assert_called_with(tf.constant([3.0]), cmap="Reds")
    ax_2.set_xticks.assert_called_with([])
    ax_2.set_yticks.assert_called_with([])
    ax_2.set_title.assert_called_with("3 - 0.30000")

    ax_3.imshow.assert_called_with(tf.constant([4.0]), cmap="Blues")
    ax_3.set_xticks.assert_called_with([])
    ax_3.set_yticks.assert_called_with([])
    ax_3.set_title.assert_called_with("foo - 0.10000")
