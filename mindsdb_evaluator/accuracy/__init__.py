# flake8: noqa  # otherwise F401 is triggered when adding all metrics from sklearn, even though no overwrites happen
import importlib
from sklearn.metrics import __all__ as __sklearn_accs__

from mindsdb_evaluator.accuracy.general import evaluate_accuracy, evaluate_accuracies
from mindsdb_evaluator.accuracy.regression import evaluate_regression_accuracy
from mindsdb_evaluator.accuracy.classification import evaluate_multilabel_accuracy, evaluate_top_k_accuracy
from mindsdb_evaluator.accuracy.forecasting import \
    evaluate_array_accuracy, \
    evaluate_num_array_accuracy, \
    evaluate_cat_array_accuracy, \
    complementary_smape_array_accuracy


__sklearn_accs__ = [acc for acc in __sklearn_accs__ if acc[0].lower() == acc[0]]

# TODO: enable custom arg passing for sklearn metrics, and to_ordinal (stateful?)
sk_metrics = importlib.import_module('sklearn.metrics')
for skacc in __sklearn_accs__:
    globals()[skacc] = getattr(sk_metrics, skacc)

__all__ = ['evaluate_accuracy', 'evaluate_accuracies', 'evaluate_regression_accuracy', 'evaluate_multilabel_accuracy',
           'evaluate_top_k_accuracy', 'evaluate_array_accuracy', 'evaluate_num_array_accuracy',
           'evaluate_cat_array_accuracy', 'complementary_smape_array_accuracy'] + __sklearn_accs__  # noqa
