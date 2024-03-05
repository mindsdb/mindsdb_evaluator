from mindsdb_evaluator.accuracy.general import evaluate_accuracy, evaluate_accuracies
from mindsdb_evaluator.accuracy.regression import evaluate_regression_accuracy
from mindsdb_evaluator.accuracy.classification import evaluate_multilabel_accuracy, evaluate_top_k_accuracy
from mindsdb_evaluator.accuracy.forecasting import \
    evaluate_array_accuracy, \
    evaluate_num_array_accuracy, \
    evaluate_cat_array_accuracy, \
    complementary_smape_array_accuracy

from sklearn.metrics import __all__ as __sklearn_accs__
__sklearn_accs__ = [acc for acc in __sklearn_accs__ if acc[0].lower() == acc[0]]

# TODO: enable custom arg passing for sklearn metrics, also need to_ordinal and back, stateful?
for skacc in __sklearn_accs__:
    exec(f'from sklearn.metrics import {skacc}')

__all__ = ['evaluate_accuracy', 'evaluate_accuracies', 'evaluate_regression_accuracy', 'evaluate_multilabel_accuracy',
           'evaluate_top_k_accuracy', 'evaluate_array_accuracy', 'evaluate_num_array_accuracy',
           'evaluate_cat_array_accuracy', 'complementary_smape_array_accuracy'] + __sklearn_accs__  # noqa
