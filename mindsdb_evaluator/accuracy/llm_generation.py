import numpy as np
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from typing import List


def evaluate_rag(metric: str,
                 predictions: List[str],
                 context: List[str],
                 question: List[str],
                 ground_truth: List[str]) -> float:
    """
    Evaluate the LLM responses using RAGAS.
    param metric: which RAGAS metric to use
    param predictions: the answers from the llm
    param context: the context used by RAG
    param question: the inputs to RAG
    param ground_truth: the ground truth for responses
    return: score
    """
    new_context = [[str(entry)] for entry in context]

    # Create the dataset expected by RAGAS
    data = {
        "question": question,
        "contexts": new_context,
        "answer": predictions,
        "ground_truth": ground_truth
    }

    # Define the feature types
    features = Features({
        "question": Value("string"),
        "contexts": Sequence(Value("string")),
        "answer": Value("string"),
        "ground_truth": Value("string")
    })

    dataset = Dataset.from_dict(data, features=features)

    # Calculate the score based on the RAGAS metric
    metrics_map = {
        'faithfulness': faithfulness,
        'answer_relevancy': answer_relevancy,
        'context_precision': context_precision,
        'context_recall': context_recall}

    ragas_score = evaluate(dataset, metrics=[metrics_map.get(metric)]).to_pandas()
    result = [score for score in ragas_score[metric] if not np.isnan(score)]
    average = sum(result) / len(result) if result else 0

    return average


