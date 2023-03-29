from sklearn.metrics import f1_score, top_k_accuracy_score


def evaluate_multilabel_accuracy(y_true, y_pred, **kwargs):
    """
    Evaluates accuracy for multilabel/tag prediction.

    :return: weighted f1 score of y_pred and ground truths.
    """
    pred_values = y_pred['prediction']
    return f1_score(y_true, pred_values, average='weighted')


def evaluate_top_k_accuracy(y_true, y_pred, **kwargs):
    """
    Evaluates the number of times where the correct label is among the top k labels

    :return: top k accuracy score of y_pred and ground truths.
    """
    pred_values = y_pred['prediction']
    k = kwargs['k'] if 'k' in kwargs else 2
    return top_k_accuracy_score(y_true, pred_values, k=k)
