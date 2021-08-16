import base64
import io
from typing import List, Dict

from distinctipy import distinctipy
import numpy as np
import PIL
import umap
from bokeh.plotting import ColumnDataSource, figure, show, output_notebook
from tqdm.auto import tqdm

from tensorflow_similarity.types import FloatTensor, IntTensor, Tensor


def tensor2images(tensor: Tensor, size: int = 64) -> List[str]:
    """Convert tensor images back to in memory images
    encoded in base 64.

    Args:
        tensor: 4D tensor that represent an image list.
        size: Image size to output in pixels. Defaults to 64.

    Returns:
        list of images encoded as base64 strings
    """

    # casting as iterating over a Tensor is slow.
    data = np.array(tensor)

    # if image provided are scaled between [0,1] then rescale
    if np.max(data) <= 1:
        data = data * 255

    # cast as int so PIL accepts its
    data = np.uint8(data)

    imgs_b64 = []
    for a in tqdm(data, desc="generating diplayabe images"):
        # if single channel, treat it as black and white
        if a.shape[-1] == 1:
            a = np.reshape(a, (a.shape[0], a.shape[1]))
            img = PIL.Image.fromarray(a, 'L')
        else:
            img = PIL.Image.fromarray(a)

        img_resized = img.resize((size, size))
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img64 = 'data:image/png;base64,%s' % str(base64.b64encode(img_bytes))[2:-1]
        imgs_b64.append(img64)

    return imgs_b64


def projector(embeddings: FloatTensor,
              labels: IntTensor = None,
              class_mapping: List[int] = None,
              images: Tensor = None,
              image_size: int = 64,
              tooltips_info: Dict[str, List[str]] = None,
              pt_size: int = 3,
              colorize: bool = True,
              pastel_factor: float = 0.1,
              plot_size: int = 600,
              active_drag: str = 'box_zoom',
              densmap: bool = True):
    """Visualize the embeddings in 2D or 3D using UMAP projection

    Args:
        embeddings: [description]
        labels: [description]
        class_mapping: [description]
        images: Images to display in tooltip on hover. Usually x_test tensor.
        pt_size: Size of the points displayed,
        image_size:
        tooltips_info:
        colorize:
        pastel_factor:
        densmap: Use UMAP dense mapper which provides better density
        estimation but is a little slower.
    """

    print("perfoming projection using UMAP")
    reducer = umap.UMAP(densmap=densmap)
    # FIXME: 2d vs 3d
    cords = reducer.fit_transform(embeddings)

    # sample id
    _idxs = [i for i in range(len(embeddings))]

    # labels?
    if labels is not None:
        # if labels are already names just use them.
        if isinstance(labels[0], str):
            _labels = labels
        else:
            _labels = [int(i) for i in labels]
    else:
        # treat each examples as its own class
        _labels = _idxs

    # class name mapping?
    if class_mapping:
        _labels_txt = [class_mapping[i] for i in _labels]
    else:
        _labels_txt = [str(i) for i in _labels]

    class_list = sorted(set(_labels_txt))
    num_classes = len(class_list)

    # generate data
    data = dict(
        id=_idxs,
        x=[i[0] for i in cords],
        y=[i[1] for i in cords],
        labels=_labels,
        labels_txt=_labels_txt,
    )

    # colors if needed
    if labels is not None and colorize:
        # generate colors
        colors = {}
        for idx, c in enumerate(distinctipy.get_colors(num_classes,
                                                       pastel_factor=pastel_factor)):
            # this is needed as labels can be strings or int or else
            cls_id  = class_list[idx]
            colors[cls_id] = distinctipy.get_hex(c)

        # map point to their color
        _colors = [colors[i] for i in _labels_txt]
        data['colors'] = _colors
    else:
        _colors = []

    # building custom tooltips
    tooltips = '<div style="border:1px solid #ABABAB">'

    if images is not None:
        imgs = tensor2images(images, image_size)
        data['imgs'] = imgs
        # have to write custom tooltip html.
        tooltips += '<center><img src="@imgs"/></center>'  # noqa

    # adding user info
    if tooltips_info:
        for k, v in tooltips_info.items():
            data[k] = v
            tooltips += "%s:@%s <br>" % (k, k)

    tooltips += 'Class:@labels_txt <br>ID:@id </div>'

    # to bokeh format
    source = ColumnDataSource(data=data)
    output_notebook()
    fig = figure(tooltips=tooltips,
                 plot_width=plot_size,
                 plot_height=plot_size,
                 active_drag=active_drag,
                 active_scroll="wheel_zoom")

    # remove grid and axis
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False

    # draw points
    if len(_colors):
        fig.circle('x', 'y', size=pt_size, color='colors', source=source)
    else:
        fig.circle('x', 'y', size=pt_size, source=source)

    # render
    show(fig, notebook_handle=True)
