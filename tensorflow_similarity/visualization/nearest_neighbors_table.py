from sklearn.neighbors import NearestNeighbors
from tensorflow_similarity.visualization.utils import filter_data, make_ordinal
import plotly.graph_objects as go
import numpy as np


def plot_nearest_neighbors_table(
        x_test,
        y_test,
        x_targets,
        y_targets,
        title="Nearest Neighbors Table",
        num_neighbors=5,
        classes=None):
    """Plot the nearest neighbors given the test and target data.

    Args:
        x_test (List[Embeddings]): The list of embeddings of test dataset.
        y_test (List[Integers]): The list of labels of test dataset.
        x_targets (List[Embeddings]): The list of embeddings of target dataset.
        y_targets (List[Integers]): The list of labels of target dataset.
        title (str, optional): Title of figure. Defaults to "Nearest Neighbors".
        num_neighbors (int): Optional, the number of neighbors shown per
            target input. Defaults to 5.
        classes (Array[String|Integers]): The array-like parameter that specify
            which classes to display, when None then we display all classes.
            Defaults to None.

    Returns:
        figure (Plotly figure): The figure object that contains the
            nearest neighbors.
    """

    # filter test and targets data if classes is specified
    if classes is not None:
        x_test, y_test, _ = filter_data(
            x_test, y_test, classes)
        x_targets, y_targets, _ = filter_data(
            x_targets, y_targets, classes)

    # sort targets for better display
    x_targets = [x for _, x in sorted(zip(y_targets, x_targets))]
    y_targets = sorted(y_targets)

    # compute nearest neighbors for each of the targets
    neighbors_database = NearestNeighbors(
        n_neighbors=num_neighbors).fit(x_test)
    distances, indices = neighbors_database.kneighbors(x_targets)

    # the number of columns is number of neighbors + 1 because we need an
    # additional column for targets.
    num_cols = num_neighbors + 1
    # the number of cells rows is 2 * number of targets because for each
    # target we need a distance row as well.
    num_rows = 2 * len(y_targets)

    # compute the header of the table, should be a list like:
    # ["target class", "1st", "2nd", ...] where "1st", "2nd", etc
    # represent the 1st, 2nd closest data point to the target
    # setting dtype to object as we want to store arbitary length strings
    header_values = np.empty(num_cols, dtype=object)
    # add <b> tags to each element to bold the header, as per documentation
    # of Plotly Table API, https://plot.ly/python/table/
    bold_template = "<b>{}<b>"
    header_values[0] = bold_template.format("Target Class")
    for i in range(1, num_cols):
        value = bold_template.format(make_ordinal(i))
        header_values[i] = value

    # Compute the cell values and colors of the table. Both the color and
    # values keyword accepts a 2D arrays (or a dataframe but due to the table
    # we want to visualize the use of dataframe is not feasible). We want a
    # table with a row of target labels followed by a row of distances
    # associated with the target labels above.
    #
    # The specification for values and color 2D arrays is the transpose
    # of python convention, meaning that each row of the 2D arrays is the
    # column for the Plotly table. For example, cells_values[i][j] specifies
    # the value in the i-th column, j-th row cell. That's the reason why
    # we have a double for loop below to package the information into the
    # required specification of Plotly table.
    #
    # Fill color is been used to distinguish rows (per targets), font color is
    # been used to distinguish incorrect nearest neighbors, and cells value is
    # been used to show the labels.
    #
    # Please see: https://screenshot.googleplex.com/sqaybDJPfGG for the table
    # we want to visualize.
    cells_values = np.empty((num_cols, num_rows), dtype=object)
    cells_fill_color = np.empty((num_cols, num_rows), dtype=object)
    cells_font_color = np.empty((num_cols, num_rows), dtype=object)

    # since we want to visually group every 2 rows together (the 1st row
    # shows the labels and the 2nd row shows the distances), we will use
    # alternate fill colors for every 2 rows to accomplish this effect,
    # see the above screenshot for reference
    alternate_fill_colors = ["lightgrey", "white"]

    # Define colors for different cell
    target_fill_color = 'paleturquoise'
    incorrect_label_color = 'red'
    correct_label_color = 'darkslategray'
    line_color = 'darkslategray'
    header_font_color = 'white'
    header_fill_color = 'royalblue'
    header_height = 40
    cell_height = 30

    # build the data, fill colors, and font colors for the table
    for i, y_target in enumerate(y_targets):
        # the sorted distances of test data points to y_target
        test_distances = distances[i]
        # the assoicated indices for test_distances
        test_indices = indices[i]
        # translate those indices into the associated labels
        test_labels = [y_test[idx] for idx in test_indices]

        # in the table we have 2 rows per target (1 for labels, 1 for distances)
        # therefore to find the correct row to update we need to scale the index
        # by a factor of 2
        label_row_idx = 2 * i
        # the distance row follows the label row
        distance_row_idx = label_row_idx + 1

        # the first column of the cells data will be the target labels
        # and the string 'distance' beneath each target
        cells_values[0][label_row_idx] = bold_template.format(y_target)
        cells_values[0][distance_row_idx] = bold_template.format("distance")

        # set the fill and font colors for the target column
        cells_fill_color[0][label_row_idx] = target_fill_color
        cells_fill_color[0][distance_row_idx] = target_fill_color
        cells_font_color[0][label_row_idx] = correct_label_color
        cells_font_color[0][distance_row_idx] = correct_label_color

        cell_fill_color = alternate_fill_colors[i % 2]

        for rank in range(num_neighbors):
            test_label = test_labels[rank]
            test_distance = '{:.4f}'.format(test_distances[rank])

            # Add one because the first column is for targets and "distance"
            col_idx = rank + 1

            cells_values[col_idx][label_row_idx] = test_label
            cells_values[col_idx][distance_row_idx] = test_distance
            cells_font_color[col_idx][label_row_idx] = correct_label_color
            cells_font_color[col_idx][distance_row_idx] = correct_label_color
            cells_fill_color[col_idx][label_row_idx] = cell_fill_color
            cells_fill_color[col_idx][distance_row_idx] = cell_fill_color

            # update the color to incorrect label color and bold the text if
            # the label is incorrect to highlight incorrect nearest neighbors
            if y_target != test_label:
                cells_font_color[col_idx][label_row_idx] = \
                    incorrect_label_color
                cells_font_color[col_idx][distance_row_idx] = \
                    incorrect_label_color
                cells_values[col_idx][label_row_idx] = bold_template.format(
                    test_label)
                cells_values[col_idx][distance_row_idx] = bold_template.format(
                    test_distance)

    # Create the table figure
    figure = go.Figure(data=go.Table(
        header=dict(
            values=header_values,
            line_color=line_color,
            fill_color=header_fill_color,
            font_color=header_font_color,
            height=header_height,
        ),
        cells=dict(
            values=cells_values,
            line_color=line_color,
            fill_color=cells_fill_color,
            font_color=cells_font_color,
            height=cell_height,
        )
    ))

    # update height and title of the table figure
    figure.update_layout(height=100 * len(y_targets), title=title)
    return figure
