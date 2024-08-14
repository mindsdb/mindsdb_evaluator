import importlib
import unittest
import os

from mindsdb_evaluator.accuracy.llm_generation import evaluate_rag


class TestLLM(unittest.TestCase):
    # @unittest.skipIf(importlib.util.find_spec('ragas') is None, "`ragas` is not available, skipping RAG tests.")
    # @unittest.skipIf(os.environ.get('OPENAI_API_KEY') is None, reason='Missing API key!')
    def test_evaluate_rag(self):
        data = {
            'question': 'What is the capital of France?',
            'answer': 'The capital of France is Paris',
            'contexts': 'Paris is the capital city of France, known for its art, fashion, and culture.',
            'ground_truth': 'The capital of France is Paris'}

        # check faithfulness
        self.assertEqual(evaluate_rag('faithfulness', data), 1.0)

        # check answer relevancy
        answer_relevancy = evaluate_rag('answer_relevancy', data)
        self.assertGreaterEqual(answer_relevancy, 0.0)
        self.assertLessEqual(answer_relevancy, 1.01)

        # check context recall
        context_recall = evaluate_rag('context_recall', data)
        self.assertGreaterEqual(context_recall, 0.0)
        self.assertLessEqual(context_recall, 1.01)

        # check context precision
        context_precision = evaluate_rag('context_precision', data)
        self.assertGreaterEqual(context_precision, 0.0)
        self.assertLessEqual(context_precision, 1.01)
