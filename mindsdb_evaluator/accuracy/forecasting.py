from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, balanced_accuracy_score


def evaluate_array_accuracy(
        true_values: np.ndarray,
        predictions: np.ndarray,
        **kwargs
) -> float:
    """
    Default time series forecasting accuracy method.
    Returns mean score over all timesteps in the forecasting horizon, as determined by the `base_acc_fn` (R2 score by default).
    """  # noqa
    base_acc_fn = kwargs.get('base_acc_fn', lambda t, p: max(0, r2_score(t, p)))

    fh = true_values.shape[1]

    aggregate = 0.0
    for i in range(fh):
        aggregate += base_acc_fn([t[i] for t in true_values], [p[i] for p in predictions])

    return aggregate / fh


def evaluate_num_array_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    Evaluate accuracy in numerical array prediction tasks.

    Scores are computed for each array index (as determined by the prediction length),
    and the final accuracy is the reciprocal of the average R2 score through all steps.
    """

    def _naive(true_values, predictions, ts_analysis):
        nan_mask = (~np.isnan(true_values)).astype(int)
        predictions *= nan_mask
        true_values = np.nan_to_num(true_values, 0.0)
        return evaluate_array_accuracy(true_values, predictions, ts_analysis=ts_analysis)

    ts_analysis = kwargs.get('ts_analysis', {})
    true_values = np.array(true_values)
    predictions = np.array(predictions)

    return _naive(true_values, predictions, ts_analysis)


def evaluate_cat_array_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    Evaluate accuracy in categorical time series forecasting tasks.

    Balanced accuracy is computed for each timestep (as determined by `timeseries_settings.horizon`),
    and the final accuracy is the reciprocal of the average score through all timesteps.
    """
    ts_analysis = kwargs.get('ts_analysis', {})

    if ts_analysis and ts_analysis['tss'].group_by:
        [true_values.pop(gby_col) for gby_col in ts_analysis['tss'].group_by]

    true_values = np.array(true_values)
    predictions = np.array(predictions)

    return evaluate_array_accuracy(true_values,
                                   predictions,
                                   ts_analysis=ts_analysis,
                                   base_acc_fn=balanced_accuracy_score)


def complementary_smape_array_accuracy(
        true_values: pd.Series,
        predictions: pd.Series,
        **kwargs
) -> float:
    """
    This metric is used in forecasting tasks. It returns ``1 - (sMAPE/2)``, where ``sMAPE`` is the symmetrical mean absolute percentage error of the forecast versus actual measurements in the time series.

    As such, its domain is 0-1 bounded.
    """  # noqa
    y_true = deepcopy(true_values)
    y_pred = deepcopy(predictions)
    tss = kwargs.get('ts_analysis', {}).get('tss', False)
    if tss and tss.group_by:
        [y_true.pop(gby_col) for gby_col in kwargs['ts_analysis']['tss'].group_by]

    # nan check
    y_true = y_true.values
    y_pred = y_pred.values
    if np.isnan(y_true).any():
        # convert all nan indexes to zero pairs that don't contribute to the metric
        nans = np.isnan(y_true)
        y_true[nans] = 0
        y_pred[nans] = 0

    smape_score = smape(y_true, y_pred)
    return 1 - smape_score / 2


def smape(y_true: np.ndarray, y_pred: np.ndarray):
    thres = 1e9
    num = abs(y_pred - y_true)
    den = (abs(y_true) + abs(y_pred)) / 2
    return min(np.average(num / den), thres)
