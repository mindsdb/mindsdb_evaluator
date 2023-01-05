import unittest
import numpy as np

from mindsdb_evaluator.accuracy.forecasting import evaluate_array_accuracy, smape


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

    def test_evaluate_smape(self):
        # random
        y_true = np.array([3, -0.5, 2, 7, 2])
        y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
        assert round(smape(y_true, y_pred), 10) == 0.5553379953

        # random
        y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
        y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
        assert round(smape(y_true, y_pred), 10) == 0.6080808081

        # large error
        y_true = np.array([[1e-4, 1e-8], [-1e-4, -1e-9], [5e-5, -5e10]])
        y_pred = np.array([[1e4, 1e8], [-1e3, 2e7], [8e5, -5e4]])
        assert round(smape(y_true, y_pred), 10) == 1.99999926

        # small error
        y_true = np.array([[0.01, 1e-5], [-1, -1e-4], [5e-5, -5e2]])
        y_pred = np.array([[0.01001, 2e-5], [-1.01, -2e-4], [6e-5, -5e2 + 1]])
        assert round(smape(y_true, y_pred), 10) == 0.2546838777

        # no error
        y_true = np.array([0.001, 0.01, -1, 1e-4])
        y_pred = np.array([0.001, 0.01, -1, 1e-4])
        assert smape(y_true, y_pred) == 0.0
