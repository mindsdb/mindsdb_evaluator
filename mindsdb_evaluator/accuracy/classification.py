from sklearn.metrics import f1_score


def evaluate_multilabel_accuracy(true_values, predictions, **kwargs):
    """
    Evaluates accuracy for multilabel/tag prediction.

    :return: weighted f1 score of predictions and ground truths.
    """
    pred_values = predictions['prediction']
    return f1_score(true_values, pred_values, average='weighted')

