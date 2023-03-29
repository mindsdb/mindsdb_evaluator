import unittest
import numpy as np
import sys

sys.path.insert(0, "../mindsdb_evaluator/accuracy")
from classification import evaluate_multilabel_accuracy, evaluate_top_k_accuracy
# from mindsdb_evaluator.accuracy.classification import evaluate_multilabel_accuracy, evaluate_top_k_accuracy


class TestClassificationAccuracies(unittest.TestCase):
    def test_evaluate_multilabel_accuracy(self):
        # vacuous case: no error
        true = np.array([1, 1, 1, 0, 0, 0])
        y_pred_true = {'prediction': true}
        self.assertEqual(evaluate_multilabel_accuracy(true, y_pred_true), 1.0)

        # random
        y_true = np.array([1, 0, 1, 0, 1, 1])
        pred = np.array([0, 0, 1, 0, 1, 1])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_multilabel_accuracy(y_true, y_pred), 0.8380952381, 10)

        # large error
        y_true = np.array([1, 1, 1, 1, 1, 1])
        pred = np.array([0, 0, 0, 0, 0, 0])
        y_pred = {'prediction': pred}
        self.assertEqual(evaluate_multilabel_accuracy(y_true, y_pred), 0.0)

        # small error
        y_true = np.array([1, 1, 1, 1, 1, 1])
        pred = np.array([0, 1, 1, 1, 1, 1])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_multilabel_accuracy(y_true, y_pred), 0.9090909091, 10)

    def test_evaluate_top_k_accuracy(self):
        # vacuous case: no error
        true = np.array([1, 1, 1, 0, 0, 0])
        y_pred_true = {'prediction': true}
        self.assertEqual(evaluate_top_k_accuracy(true, y_pred_true, k=1), 1.0)

        # random
        y_true = np.array([0, 1, 2, 2])
        pred = np.array([[0.5, 0.2, 0.2],
                        [0.3, 0.4, 0.2],
                        [0.2, 0.4, 0.3],
                        [0.7, 0.2, 0.1]])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_top_k_accuracy(y_true, y_pred), 0.75)

        # large error
        y_true = np.array([0, 1, 2, 3])
        pred = np.array([[0.15, 0.5, 0.3, 0.05],
                        [0.3, 0.1, 0.3, 0.3],
                        [0.3, 0.3, 0.1, 0.3],
                        [0.3, 0.3, 0.3, 0.1]])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_top_k_accuracy(y_true, y_pred, k=3), 0.25)

        # small error
        y_true = np.array([0, 1, 2, 3, 3])
        pred = np.array([[0.15, 0.5, 0.3, 0.05],
                        [0.3, 0.3, 0.1, 0.3],
                        [0.3, 0.3, 0.3, 0.1],
                        [0.3, 0.3, 0.3, 0.1],
                        [0.1, 0.3, 0.3, 0.3]])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_top_k_accuracy(y_true, y_pred, k=3), 0.80)

