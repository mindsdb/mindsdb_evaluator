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
                      accuracy_functions: List[str],
                      ts_analysis: Optional[dict] = {},
                      n_decimals: Optional[int] = 3) -> Dict[str, float]:
    """
    Dispatcher for accuracy evaluation.

    :param data: original dataframe.
    :param predictions: output of a machine learning model for the input `data`.
    :param target: target column name.
    :param accuracy_functions: list of accuracy function names. Support currently exists for `scikit-learn`'s `metrics` module, plus any custom methods that Lightwood exposes.
    :param ts_analysis: `lightwood.data.timeseries_analyzer` output, used to compute time series task accuracy.
    :param n_decimals: used to round accuracies.
    :return: accuracy metric for a dataset and predictions.
    """  # noqa
    score_dict = {}

    for accuracy_function_str in accuracy_functions:
        if 'array_accuracy' in accuracy_function_str or accuracy_function_str in ('bounded_ts_accuracy',):
            if ts_analysis is None or not ts_analysis.get('tss', False) or not ts_analysis['tss'].is_timeseries:
                # normal array, needs to be expanded
                cols = [target]
                true_values = data[cols].apply(lambda x: pd.Series(x[target]), axis=1)
            else:
                horizon = 1 if not isinstance(predictions.iloc[0], list) else len(predictions.iloc[0])
                gby = ts_analysis.get('tss', {}).group_by if ts_analysis.get('tss', {}).group_by else []
                cols = [target] + [f'{target}_timestep_{i}' for i in range(1, horizon)] + gby
                true_values = data[cols]

            true_values = true_values.reset_index(drop=True)
            predictions = predictions.apply(pd.Series).reset_index(drop=True)  # split array cells into columns

            if accuracy_function_str == 'evaluate_array_accuracy':
                acc_fn = evaluate_array_accuracy
            elif accuracy_function_str == 'evaluate_num_array_accuracy':
                acc_fn = evaluate_num_array_accuracy
            elif accuracy_function_str == 'evaluate_cat_array_accuracy':
                acc_fn = evaluate_cat_array_accuracy
            elif accuracy_function_str == 'complementary_smape_array_accuracy':
                acc_fn = complementary_smape_array_accuracy
            else:
                raise Exception(f"Could not retrieve accuracy function: {accuracy_function_str}")

            score_dict[accuracy_function_str] = acc_fn(true_values,
                                                       predictions,
                                                       data=data[cols],
                                                       ts_analysis=ts_analysis)
        else:
            true_values = data[target].tolist()
            if hasattr(importlib.import_module('mindsdb_evaluator.accuracy'), accuracy_function_str):
                accuracy_function = getattr(importlib.import_module('mindsdb_evaluator.accuracy'),
                                            accuracy_function_str)
            else:
                accuracy_function = getattr(importlib.import_module('sklearn.metrics'), accuracy_function_str)
            score_dict[accuracy_function_str] = accuracy_function(list(true_values), list(predictions))

    for fn, score in score_dict.items():
        score_dict[fn] = round(score, n_decimals)

    return score_dict
