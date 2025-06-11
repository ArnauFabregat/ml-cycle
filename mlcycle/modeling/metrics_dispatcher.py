import pandas
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict


def metric(metric_name: str, params: Dict, y_true: pandas.Series, y_pred: pandas.Series) -> float:
    """
    Compute the specified evaluation metric given the true and predicted labels.

    Args:
        metric_name: The name of the evaluation metric to compute.
        params: A dictionary containing additional parameters required for the metric computation.
        y_true: The true labels or target values.
        y_pred: The predicted labels or target values.

    Returns:
        The computed score for the evaluation metric.

    Raises:
        ValueError: If the specified metric_name is not supported.
    """
    # Set the true and predicted labels in the parameters dictionary
    params['y_true'] = y_true
    params['y_pred'] = y_pred

    # Classification metrics
    if metric_name == 'fbeta_score':
        score = fbeta_score(**params)

    elif metric_name == 'accuracy_score':
        score = accuracy_score(**params)

    elif metric_name == 'recall_score':
        score = recall_score(**params)

    elif metric_name == 'precision_score':
        score = precision_score(**params)

    # Regression metrics
    elif metric_name == 'mean_absolute_percentage_error':
        score = mean_absolute_percentage_error(**params)

    elif metric_name == 'mean_squared_error':
        score = mean_squared_error(**params)

    elif metric_name == 'mean_absolute_error':
        score = mean_absolute_error(**params)

    elif metric_name == 'r2_score':
        score = r2_score(**params)

    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    return score
