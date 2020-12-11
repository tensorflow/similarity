from tensorflow_similarity.metrics import precision, recall, f1_score


def test_precision():
    y = [1, 1, 1, 1]
    same = [1, 1, 1, 1]
    diff = [2, 3, 4, 4]
    half = [1, 3, 1, 2]
    assert precision(y, same) == 1
    assert precision(y, diff) == 0
    assert precision(y, half) == 0.5

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    assert round(float(precision(y_true, y_pred)), 2) == 0.33


def test_recall():
    y = [2, 1, 3, 1]
    same = [2, 1, 3, 1]
    diff = [3, 2, 1, 2]
    half = [2, 3, 3, 2]
    assert recall(y, same) == 1
    assert recall(y, diff) == 0
    assert recall(y, half) == 0.5

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    assert round(float(recall(y_true, y_pred)), 2) == 0.33


def test_f1_score():
    # from sklearn:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]

    assert round(float(f1_score(y_true, y_pred)), 2) == 0.33

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    assert round(float(f1_score(y_true, y_pred)), 2) == 1
