from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, balanced_accuracy_score


def evaluate_array_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
) -> float:
    """
    Default forecasting accuracy metric.
    
    Yields mean score over all timesteps in the forecast, as determined by the `base_acc_fn` (R2 score by default).
    """  # noqa

    base_acc_fn = kwargs.get('base_acc_fn', lambda t, p: max(0, r2_score(t, p)))
    fh = y_true.shape[1]
    aggregate = 0.0
    for i in range(fh):
        aggregate += base_acc_fn([t[i] for t in y_true], [p[i] for p in y_pred])

    return aggregate / fh


def evaluate_num_array_accuracy(
        y_true: pd.Series,
        y_pred: pd.Series,
        **kwargs
) -> float:
    """
    Numerical forecast accuracy metric.

    Scores are computed for each array index (as determined by the forecast length),
    and the final accuracy is the reciprocal of the average R2 score through all steps.
    """  # noqa

    def _naive(yt: np.ndarray, yp: np.ndarray, _ts_analysis: dict):
        nan_mask = (~np.isnan(yt)).astype(int)
        yp *= nan_mask
        yt = np.nan_to_num(yt, nan=0.0)
        return evaluate_array_accuracy(yt, yp, ts_analysis=_ts_analysis)

    ts_analysis = kwargs.get('ts_analysis', {})
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return _naive(y_true, y_pred, ts_analysis)


def evaluate_cat_array_accuracy(
        y_true: pd.Series,
        y_pred: pd.Series,
        **kwargs
) -> float:
    """
    Categorical forecast accuracy metric.

    Balanced accuracy is computed for each timestep (as determined by the forecast length)
    and the final accuracy is the reciprocal of the average score through all timesteps.
    """  # noqa
    ts_analysis = kwargs.get('ts_analysis', {})

    if ts_analysis and ts_analysis['tss'].group_by:
        [y_true.pop(gby_col) for gby_col in ts_analysis['tss'].group_by]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return evaluate_array_accuracy(y_true,
                                   y_pred,
                                   ts_analysis=ts_analysis,
                                   base_acc_fn=balanced_accuracy_score)


def complementary_smape_array_accuracy(
        y_true: pd.Series,
        y_pred: pd.Series,
        **kwargs
) -> float:
    """
    Forecast accuracy metric. 
    
    It returns ``1 - (sMAPE/2)``, where ``sMAPE`` is the symmetrical mean absolute percentage error of the forecast versus actual measurements in the time series.

    As such, its domain is 0-1 bounded.
    """  # noqa
    y_true = deepcopy(y_true)
    y_pred = deepcopy(y_pred)
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
    """ Symmetrical mean absolute percentage error. """  # noqa
    thres = 1e9
    num = abs(y_pred - y_true)
    den = (abs(y_true) + abs(y_pred)) / 2
    return min(np.average(num / den), thres)
