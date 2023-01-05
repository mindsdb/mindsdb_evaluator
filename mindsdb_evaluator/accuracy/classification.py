from sklearn.metrics import f1_score


def evaluate_multilabel_accuracy(y_true, y_pred, **kwargs):
    """
    Evaluates accuracy for multilabel/tag prediction.

    :return: weighted f1 score of y_pred and ground truths.
    """
    pred_values = y_pred['prediction']
    return f1_score(y_true, pred_values, average='weighted')
