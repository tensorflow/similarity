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

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import umap
from MulticoreTSNE import MulticoreTSNE as multiTSNE
from tensorflow_similarity.visualization.utils import filter_data


def plot_embedding_projector(
        x_data,
        y_labels,
        title="Embedding Projector",
        num_steps=500,
        use_umap=False,
        interactive=True,
        classes=None):
    """Plot the embedding projector of the given dataset.

    Args:
        x_data (List[Embeddings]): The list of embeddings.
        y_labels (List[Integers]): The list of labels. Should be the same
            length as x_data and y_labels[i] is the label for x_data[i].
        title (str, optional): Title of figure. Defaults to "Nearest Neighbors".
        num_steps (Integer, optional): The number of steps to train the
            2D embedding. For TSNE this parameter corresponds to n_iter, for
            umap it corresponds to n_epochs. Defaults to 500.
        use_umap (bool, optional): When True we will use umap technique to
            embed the data into a 2-dimensional space. Otherwise use tsne.
            Defaults to False.
        interactive (bool, optional): When set to true, we will return an
            interactive plot using Altair, otherwise return a static plot.
            Defaults to True.
        classes (Array[String|Integers]): The array-like parameter that specify
            which classes to display, when None then we display all classes.
            Defaults to None.

    Returns:
        figure (matplotlib figure | PlotlyAltair chart): The figure object that
            contains the embedding projector plot.
    """

    # filter the data if classes is specified
    if classes is not None:
        x_data, y_labels, _ = filter_data(
            x_data, y_labels, classes)

    # initizlie embedding method
    if use_umap:
        embedding_method = umap.UMAP(
            n_components=2,
            n_neighbors=10,
            min_dist=0.001,
            n_epochs=num_steps)
    else:
        # try to import tsne-cuda if available for speed and scalability
        # put this import inside the function as to not slow down the import
        # of the entire package. This is ~70X speedup compares to MulticoreTSNE
        # and 650X speedup compares to sklearn's TSNE
        # Reference: https://github.com/CannyLab/tsne-cuda
        try:
            import tsnecuda
            tsnecuda.test()
            from tsnecuda import TSNE
        except BaseException:
            TSNE = multiTSNE

        embedding_method = TSNE(n_components=2, n_iter=num_steps)

    # compute the 2D embedding
    x_embedded = embedding_method.fit_transform(x_data)

    # turn the embedding into a dataframe
    data = dict()
    data["x"] = x_embedded[:, 0]
    data["y"] = x_embedded[:, 1]
    data["class"] = y_labels

    data_df = pd.DataFrame(data)
    data_df["class"] = data_df["class"].apply(str)

    if interactive:
        height = 350
        width = 350

        # interactive components
        brush = alt.selection(type='interval')
        selection = alt.selection_multi(fields=['class'])

        # scatter plot component
        scatter_color = alt.condition(
            brush, 'class', alt.value('lightgray'), legend=None)
        scatter = alt.Chart(data_df).mark_point().encode(
            x='x',
            y='y',
            color=scatter_color,
        ).add_selection(
            brush
        ).transform_filter(
            selection
        ).properties(
            width=width, height=height
        )

        # bar chart component
        bars_color = alt.condition(
            selection,
            'class',
            alt.value('lightgray'),
            legend=None)
        bars = alt.Chart(data_df).mark_bar().encode(
            x='class',
            y='count(class)',
            color=bars_color
        ).transform_filter(
            brush
        ).add_selection(
            selection
        ).properties(
            width=width, height=height
        )

        # selectable legend component
        legend_color = alt.condition(
            selection,
            'class',
            alt.value('lightgray'),
            legend=None)
        legend = alt.Chart(data_df).mark_point(filled=True, size=300).encode(
            y=alt.Y('class', axis=alt.Axis(orient='right')),
            color=legend_color
        ).add_selection(
            selection
        ).properties(
            height=height,
            width=30
        )

        # horizontal concatenate components
        figure = scatter | bars | legend

        # add title to the figure
        figure = figure.properties(title=title)
    else:
        figure, ax = plt.subplots()
        sns.scatterplot(
            data_df["x"],
            data_df["y"],
            hue=data_df["class"],
            palette=sns.color_palette(
                "husl",
                10),
            s=20)
        ax.set_title(title)
        # Remove ticks and put the legend out of the figure
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)

    return figure
