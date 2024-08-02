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

def evaluate_faithfulness(predictions: List[str], 
                          context: List[str], 
                          question: List[str],
                          ground_truth: List[str]) -> float:
    """
    Evaluate the faithfulness of LLM responses using RAGAS.
    :return: Faithfulness score.
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
    faithfulness_score = evaluate(dataset, metrics=[faithfulness, context_recall, context_precision, answer_relevancy])

    result = faithfulness_score.to_pandas()
    print('result', result)
    print(result.columns)
    scores_dict = {}
    for column in result.columns:
        if column in ["faithfulness", "context_recall", "context_precision", "answer_relevancy"]:

            score = result[column].to_list()
            filtered_scores = [score for score in score if not np.isnan(score)]
            average = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0
            scores_dict[column] = average

    return scores_dict


