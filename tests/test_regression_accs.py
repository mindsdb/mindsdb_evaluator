import unittest
import numpy as np

from mindsdb_evaluator.accuracy.regression import evaluate_regression_accuracy


class TestRegressionAccuracies(unittest.TestCase):
    def test_evaluate_regression_accuracy(self):
        # vacuous case: with complete range provided
        true = np.array([1,2,3])
        pred_true = {'lower': np.min(true), 'upper': np.max(true)}
        self.assertEqual(evaluate_regression_accuracy(true, pred_true), 1.0)

        # vacuous case: without complete range provided
        true = np.array([1,2,3])
        pred_true = {'prediction': true}
        self.assertEqual(evaluate_regression_accuracy(true, pred_true), 1.0)

        # edge case: y_pred range provided falls outside of the span of y_true
        y_true = np.array([100.0,200.0,300.0])
        pred = np.array([0.0, 10.0, 20.0])
        y_pred = {'lower': np.min(pred), 'upper': np.max(pred)}
        self.assertEqual(evaluate_regression_accuracy(y_true, y_pred), 0.0)

        # edge case: y_pred range provided falls outside of the span of y_true
        y_true = np.array([100.0, 200.0, 300.0])
        pred = np.array([1000.0, 2000.0, 3000.0])
        y_pred = {'lower': np.min(pred), 'upper': np.max(pred)}
        self.assertEqual(evaluate_regression_accuracy(y_true, y_pred), 0.0)

        # random predictions in range
        y_true = np.array([0.5, -3.0, 4.9])
        pred = np.array([-3.2, -5.0, -2.3])
        y_pred = {'lower': np.min(pred), 'upper': np.max(pred)}
        self.assertAlmostEqual(evaluate_regression_accuracy(y_true, y_pred), 0.3333333333333333, 10)

        # random
        y_true = np.array([0.97939651, 0.24582314, 0.65548829])
        pred = np.array([0.76994255, 0.26130589, 0.44479124])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_regression_accuracy(y_true, y_pred), 0.6725601763, 10)

        # large error
        y_true = np.array([0.75, 0.85, 0.25])
        pred = np.array([0.55, 0.65, -0.05])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_regression_accuracy(y_true, y_pred), 0.1774193548, 10)

        # small error
        y_true = np.array([3, -0.5, 2])
        pred = np.array([2.5, 0.0, 2])
        y_pred = {'prediction': pred}
        self.assertAlmostEqual(evaluate_regression_accuracy(y_true, y_pred), 0.9230769231, 10)

        # edge case
        y_true = np.array([1,2,3])
        pred = np.array([3,2,1])
        y_pred = {'prediction': pred}
        self.assertEqual(evaluate_regression_accuracy(y_true, y_pred), 0.0)