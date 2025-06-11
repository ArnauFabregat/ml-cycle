from mlcycle.modeling.metrics_dispatcher import metric
import numpy
import pandas
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from typing import Dict


def cross_validation_technique(cv_name: str, params: Dict,
                               X: pandas.DataFrame, y: pandas.DataFrame,
                               model: Pipeline, metric_dict: Dict) -> float:
    """
    Perform cross-validation using the specified cross-validation technique and compute evaluation metrics.

    Args:
        cv_name: The name of the cross-validation technique to use.
        params: A dictionary containing additional parameters required for the cross-validation technique.
        X: The feature matrix.
        y: The target variable.
        model: The machine learning model or pipeline to use.
        metric_dict: A dictionary containing the evaluation metric details.

    Returns:
        The computed score result.

    Raises:
        ValueError: If the specified cv_name is not supported.
    """

    if cv_name == 'StratifiedKFold':
        # Build Cross-Validator
        cv = StratifiedKFold(**params)
        tr_scores = []
        ts_scores = []

        # Calculate metrics for the different folds
        for _, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_cv_tr, X_cv_ts = X.iloc[train_idx], X.iloc[test_idx]
            y_cv_tr, y_cv_ts = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_cv_tr, y_cv_tr)

            # Get train metrics
            tr_pred = model.predict(X_cv_tr)
            tr_score = metric(metric_name=metric_dict['name'],
                              y_true=y_cv_tr,
                              y_pred=tr_pred,
                              params=metric_dict['params'])
            tr_scores.append(tr_score)

            # Get test metrics
            ts_pred = model.predict(X_cv_ts)
            ts_score = metric(metric_name=metric_dict['name'],
                              y_true=y_cv_ts,
                              y_pred=ts_pred,
                              params=metric_dict['params'])
            ts_scores.append(ts_score)

        tr_avg_score = numpy.mean(tr_scores)
        ts_avg_score = numpy.mean(ts_scores)

    else:
        raise ValueError(f"Unsupported cv_name: {cv_name}")

    # Final score with penalization
    score_result = ts_avg_score * (1-abs(tr_avg_score - ts_avg_score))

    return score_result
