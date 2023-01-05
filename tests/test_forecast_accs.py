import unittest
import numpy as np

from mindsdb_evaluator.accuracy.forecasting import evaluate_array_accuracy


class TestForecastingAccuracies(unittest.TestCase):
    def test_evaluate_array_r2_accuracy(self):
        true = np.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]])
        self.assertTrue(evaluate_array_accuracy(true, true) == 1.0)

        pred = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.assertTrue(evaluate_array_accuracy(true, pred) == 0.0)

        pred = np.array([[i + 1 for i in instance] for instance in true])
        self.assertGreaterEqual(evaluate_array_accuracy(true, pred), 0.99)

        pred = np.array([[i - 1 for i in instance] for instance in true])
        self.assertGreaterEqual(evaluate_array_accuracy(true, pred), 0.99)

        pred = np.array([[-i for i in instance] for instance in true])
        self.assertTrue(evaluate_array_accuracy(true, pred) == 0.0)
