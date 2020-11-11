from tensorflow_similarity.indexer import Indexer


def calibrate(model, x, y):
    """Calibrate the model to provide a uniform scoring for matching

    Args:
        x ([type]): [description]
        y ([type]): [description]
    Returns:
        dict: Calibration score
    """
    indx = Indexer()

    # adding our embedding and metadata to index. Note: labels and data a
    # are optional
    indx.batch_add(embeddings, labels, data)

    # building the index: this is needed after points are added to the index
    indx.build()
