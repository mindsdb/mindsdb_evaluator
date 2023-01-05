import importlib
from typing import List, Dict, Optional

import pandas as pd

from mindsdb_evaluator.accuracy.forecasting import \
    evaluate_array_accuracy, \
    evaluate_num_array_accuracy, \
    evaluate_cat_array_accuracy, \
    complementary_smape_array_accuracy


def evaluate_accuracy(data: pd.DataFrame,
                      predictions: pd.Series,
                      target: str,
                      accuracy_function: str,
                      ts_analysis: Optional[dict] = {},
                      n_decimals: Optional[int] = 3) -> float:
    """
    Dispatcher for accuracy evaluation.

    :param data: original dataframe.
    :param predictions: output of a machine learning model for the input `data`.
    :param target: target column name.
    :param accuracy_function: either a metric from the `accuracy` module or `scikit-learn.metric`.
    :param ts_analysis: `lightwood.data.timeseries_analyzer` output, used to compute time series task accuracy.
    :param n_decimals: used to round accuracies.
    
    :return: accuracy score, given input data and model predictions.
    """  # noqa
    if 'array_accuracy' in accuracy_function or accuracy_function in ('bounded_ts_accuracy',):
        if ts_analysis is None or not ts_analysis.get('tss', False) or not ts_analysis['tss'].is_timeseries:
            # normal array, needs to be expanded
            cols = [target]
            y_true = data[cols].apply(lambda x: pd.Series(x[target]), axis=1)
        else:
            horizon = 1 if not isinstance(predictions.iloc[0], list) else len(predictions.iloc[0])
            gby = ts_analysis.get('tss', {}).group_by if ts_analysis.get('tss', {}).group_by else []
            cols = [target] + [f'{target}_timestep_{i}' for i in range(1, horizon)] + gby
            y_true = data[cols]

        y_true = y_true.reset_index(drop=True)
        y_pred = predictions.apply(pd.Series).reset_index(drop=True)  # split array cells into columns

        if accuracy_function == 'evaluate_array_accuracy':
            acc_fn = evaluate_array_accuracy
        elif accuracy_function == 'evaluate_num_array_accuracy':
            acc_fn = evaluate_num_array_accuracy
        elif accuracy_function == 'evaluate_cat_array_accuracy':
            acc_fn = evaluate_cat_array_accuracy
        elif accuracy_function == 'complementary_smape_array_accuracy':
            acc_fn = complementary_smape_array_accuracy
        else:
            raise Exception(f"Could not retrieve accuracy function: {accuracy_function}")

        score = acc_fn(y_true, y_pred, data=data[cols], ts_analysis=ts_analysis)
    else:
        y_true = data[target].tolist()
        y_pred = list(predictions)
        if hasattr(importlib.import_module('mindsdb_evaluator.accuracy'), accuracy_function):
            accuracy_function = getattr(importlib.import_module('mindsdb_evaluator.accuracy'),
                                        accuracy_function)
        else:
            accuracy_function = getattr(importlib.import_module('sklearn.metrics'), accuracy_function)

        score = accuracy_function(y_true, y_pred)

    return round(score, n_decimals)


def evaluate_accuracies(data: pd.DataFrame,
                        predictions: pd.Series,
                        target: str,
                        accuracy_functions: List[str],
                        ts_analysis: Optional[dict] = {},
                        n_decimals: Optional[int] = 3) -> Dict[str, float]:
    score_dict = {}
    for accuracy_function in accuracy_functions:
        score = evaluate_accuracy(data, predictions, target, accuracy_function, ts_analysis, n_decimals)
        score_dict[accuracy_function] = score

    return score_dict
