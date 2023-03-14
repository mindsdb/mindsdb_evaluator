import unittest
import numpy as np

from mindsdb_evaluator.accuracy.classification import evaluate_multilabel_accuracy


class TestClassificationAccuracies(unittest.TestCase):
    def test_evaluate_multilabel_accuracy(self):
        # vacuous case: no error
        true = np.array([1,1,1,0,0,0])
        y_pred_true = {'prediction': true}
        self.assertEqual(evaluate_multilabel_accuracy(true, y_pred_true), 1.0)

        # random
        y_true = np.array([1,0,1,0,1,1])
        pred = np.array([0,0,1,0,1,1])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_multilabel_accuracy(y_true, y_pred), 0.8380952381, 10)

        # large error
        y_true = np.array([1,1,1,1,1,1])
        pred = np.array([0,0,0,0,0,0])
        y_pred = {'prediction': pred}
        self.assertEqual(evaluate_multilabel_accuracy(y_true, y_pred), 0.0)

        # small error
        y_true = np.array([1,1,1,1,1,1])
        pred = np.array([0,1,1,1,1,1])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_multilabel_accuracy(y_true, y_pred), 0.9090909091, 10)