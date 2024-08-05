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
    :return: score
    """

    # Create the dataset expected by RAGAS
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

    # Calculate the faithfulness score
    if metric == 'faithfulness':
        ragas_metric = faithfulness
    elif metric == 'answer_relevancy':
        ragas_metric = answer_relevancy
    elif metric == 'context_precision':
        ragas_metric = context_precision
    elif metric == 'context_recall':
        ragas_metric = context_recall
    ragas_score = evaluate(dataset, metrics=[ragas_metric])

    result = ragas_score.to_pandas()
    filtered_scores = [score for score in result[metric] if not np.isnan(score)]
    average = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0

    return average


