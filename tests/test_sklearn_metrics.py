import unittest
import pandas as pd
from mindsdb_evaluator import evaluate_accuracy, evaluate_accuracies


class TestSklearnMetrics(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame.from_records({'target': [0, 1, 2, 0, 1, 2]})
        self.pred = pd.Series([0, 2, 1, 0, 0, 1])
        self.target = 'target'

    # PRECISION SCORE #
    def test_sklearn_precision_score_macro(self):
        acc = evaluate_accuracy(self.data, self.pred, 'precision_score', self.target,
                                fn_kwargs={'average': 'macro'})
        self.assertAlmostEqual(acc, 0.222)

    def test_sklearn_precision_score_micro(self):
        acc = evaluate_accuracy(self.data, self.pred, 'precision_score', self.target,
                                fn_kwargs={'average': 'micro'})
        self.assertAlmostEqual(acc, 0.333)

    def test_sklearn_precision_score_weighted(self):
        acc = evaluate_accuracy(self.data, self.pred, 'precision_score', self.target,
                                fn_kwargs={'average': 'weighted'})
        self.assertAlmostEqual(acc, 0.222)

    def test_sklearn_precision_score_none(self):
        expected_err_str = "Accuracy function `precision_score` returned invalid type *"
        with self.assertRaisesRegex(AssertionError, expected_err_str):
            evaluate_accuracy(self.data, self.pred, 'precision_score', self.target,
                              fn_kwargs={'average': None})

    def test_sklearn_precision_score_binary_for_multiclass(self):
        expected_err_str = "Target is multiclass but average='binary'.*"
        with self.assertRaisesRegex(ValueError, expected_err_str):
            evaluate_accuracy(self.data, self.pred, 'precision_score', self.target,
                              fn_kwargs={'average': 'binary'})

    # RECALL SCORE #
    def test_sklearn_recall_score_macro(self):
        acc = evaluate_accuracy(self.data, self.pred, 'recall_score', self.target,
                                fn_kwargs={'average': 'macro'})
        self.assertAlmostEqual(acc, 0.333)

    def test_sklearn_recall_score_micro(self):
        acc = evaluate_accuracy(self.data, self.pred, 'recall_score', self.target,
                                fn_kwargs={'average': 'micro'})
        self.assertAlmostEqual(acc, 0.333)

    def test_sklearn_recall_score_weighted(self):
        acc = evaluate_accuracy(self.data, self.pred, 'recall_score', self.target,
                                fn_kwargs={'average': 'weighted'})
        self.assertAlmostEqual(acc, 0.333)

    def test_sklearn_recall_score_none(self):
        expected_err_str = "Accuracy function `recall_score` returned invalid type *"
        with self.assertRaisesRegex(AssertionError, expected_err_str):
            evaluate_accuracy(self.data, self.pred, 'recall_score', self.target,
                              fn_kwargs={'average': None})

    def test_sklearn_recall_score_binary_for_multiclass(self):
        expected_err_str = "Target is multiclass but average='binary'.*"
        with self.assertRaisesRegex(ValueError, expected_err_str):
            evaluate_accuracy(self.data, self.pred, 'recall_score', self.target,
                              fn_kwargs={'average': 'binary'})

    def test_sklearn_wrapped_with_args(self):
        acc_fn = 'recall_score'
        acc_fns = [{"module": acc_fn, "args": {'average': 'macro'}}]
        acc = evaluate_accuracies(self.data, self.pred, self.target, acc_fns)
        print(acc)
        self.assertAlmostEqual(acc[acc_fn], 0.333)
