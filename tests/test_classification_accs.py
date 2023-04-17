import unittest
import numpy as np
from mindsdb_evaluator.accuracy.classification import evaluate_multilabel_accuracy, evaluate_top_k_accuracy
from sklearn.metrics import top_k_accuracy_score


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
        ground_accuracy = top_k_accuracy_score(y_true=true, y_score=true, k=1, normalize=True)
        module_accuracy = evaluate_top_k_accuracy(true, y_pred_true, k=1)
        self.assertEqual(module_accuracy, 1.0)
        self.assertEqual(module_accuracy, ground_accuracy)

        # random
        y_true = np.array([0, 1, 2, 2])
        pred = np.array([[0.5, 0.2, 0.2],
                        [0.3, 0.4, 0.2],
                        [0.2, 0.4, 0.3],
                        [0.7, 0.2, 0.1]])
        y_pred = {'prediction': pred}
        ground_accuracy = top_k_accuracy_score(y_true=y_true, y_score=pred, k=2, normalize=True)
        module_accuracy = evaluate_top_k_accuracy(y_true, y_pred)
        self.assertAlmostEqual(module_accuracy, 0.75, 10)
        self.assertEqual(module_accuracy, ground_accuracy)

        # large error
        y_true = np.array([0, 1, 2, 3])
        pred = np.array([[0.15, 0.5, 0.3, 0.05],
                        [0.3, 0.1, 0.3, 0.3],
                        [0.3, 0.3, 0.1, 0.3],
                        [0.3, 0.3, 0.3, 0.1]])
        y_pred = {'prediction': pred}
        ground_accuracy = top_k_accuracy_score(y_true=y_true, y_score=pred, k=3, normalize=True)
        module_accuracy = evaluate_top_k_accuracy(y_true, y_pred, k=3)
        self.assertAlmostEqual(module_accuracy, 0.25, 10)
        self.assertEqual(module_accuracy, ground_accuracy)

        # small error
        y_true = np.array([0, 1, 2, 3, 3])
        pred = np.array([[0.15, 0.5, 0.3, 0.05],
                        [0.3, 0.3, 0.1, 0.3],
                        [0.3, 0.3, 0.3, 0.1],
                        [0.3, 0.3, 0.3, 0.1],
                        [0.1, 0.3, 0.3, 0.3]])
        y_pred = {'prediction': pred}
        ground_accuracy = top_k_accuracy_score(y_true=y_true, y_score=pred, k=3, normalize=True)
        module_accuracy = evaluate_top_k_accuracy(y_true, y_pred, k=3)
        self.assertAlmostEqual(module_accuracy, 0.8, 10)
        self.assertEqual(module_accuracy, ground_accuracy)
