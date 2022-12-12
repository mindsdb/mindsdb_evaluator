import numpy as np
from sklearn.metrics import r2_score


def evaluate_regression_accuracy(
        true_values,
        predictions,
        **kwargs
):
    """
    Evaluates accuracy for regression tasks.
    If predictions have a lower and upper bound, then `within-bound` accuracy is computed: whether the ground truth value falls within the predicted region.
    If not, then a (positive bounded) R2 score is returned instead.

    :return: accuracy score as defined above. 
    """  # noqa
    if 'lower' and 'upper' in predictions:
        Y = np.array(true_values).astype(float)
        within = ((Y >= predictions['lower']) & (Y <= predictions['upper']))
        return sum(within) / len(within)
    else:
        r2 = r2_score(true_values, predictions['prediction'])
        return max(r2, 0)

