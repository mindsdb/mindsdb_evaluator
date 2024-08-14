import numpy as np
import pandas as pd


def _setup_rag_eval(data: pd.DataFrame, predictions: pd.Series) -> dict:
    """
    param data: input data for RAGAS
    param predictions: answers from the LLM
    return: dict of the fields in the correct format
    """
    context = list(data['contexts'])
    question = list(data['question'])
    ground_truth = list(data['ground_truth'])
    answer = predictions.tolist()
    return {'contexts': context, 'question': question, 'ground_truth': ground_truth, 'answer': answer}


def evaluate_rag(metric: str,
                 fields: dict) -> float:
    """
    Evaluate the LLM responses using RAGAS.
    param metric: which RAGAS metric to use
    param fields: contains the input fields for RAGAS
    return: score
    """
    try:
        # ugly workaround, see `pyproject.toml` for why
        from datasets import Dataset, Features, Sequence, Value
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
    except Exception:
        raise Exception("Dependencies for RAG metrics are not available. Re-install using `pip install mindsdb-evaluator[rag] and try again.")  # noqa

    new_context = [[str(entry)] for entry in [fields['contexts']]]

    try:
        data = {
            "question": [fields['question']],
            "contexts": new_context,
            "answer": [fields['answer']],
            "ground_truth": [fields['ground_truth']],
        }
    except KeyError:
        raise Exception(f'''To evaluate `{metric}`, check that your input data contains
                        `question`, `context`, `ground_truth` and try again''')

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


