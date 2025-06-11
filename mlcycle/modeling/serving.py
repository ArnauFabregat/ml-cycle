from mlcycle.feature_engineering.outliers_dispatcher import Outliers
from mlcycle.feature_engineering.imputers_dispatcher import Imputers
from mlcycle.modeling.estimators_dispatcher import Classifiers
from mlcycle.modeling.metrics_dispatcher import metric
import pandas
from sklearn.pipeline import Pipeline
from typing import Dict, Callable


def fit_model(pipeline_config: Dict,
              X: pandas.DataFrame,
              y: pandas.DataFrame,
              save: bool = False,
              path: str = None) -> Callable:
    """
    Fit a machine learning model using a configurable pipeline.

    Args:
        pipeline_config: A dictionary containing configuration settings for the pipeline.
        X: The input features.
        y: The target variable.
        save: Whether to save the trained model.
        path: The path where the trained model should be saved.

    Returns:
        model_instance: The trained machine learning model.
    """
    # TODO add classifiers/regressors condition

    # Extract configuration settings from the pipeline_config dictionary
    dict_outliers = pipeline_config['features']['outliers_']
    dict_imputers = pipeline_config['features']['imputer_']
    estimator_dict = pipeline_config['estimator']

    # Create a preprocessing pipeline
    outliers_handler_instance = Outliers(dict_outliers).run()
    impute_instance = Imputers(dict_imputers).run()

    pipeline_preprocess = Pipeline(
            [
                ('outliers_handler', outliers_handler_instance),
                ('imputation', impute_instance)
            ]
    )

    # Apply preprocessing to the input data
    x_train_eval = pipeline_preprocess.fit_transform(X)

    # Create and train the machine learning model
    model_instance = Classifiers(estimator_dict).run()
    model_instance.fit(x_train_eval, y)

    if save:
        # TODO: Implement model saving logic
        print('TODO: Implement model saving logic')

    return model_instance


def train_test_eval(pipeline_config: Dict,
                    X_train: pandas.DataFrame,
                    y_train: pandas.DataFrame,
                    X_test: pandas.DataFrame,
                    y_test: pandas.DataFrame) -> None:
    """
    Train a machine learning model, evaluate it on test data, and print the evaluation metric.

    Args:
        pipeline_config: A dictionary containing configuration settings for the pipeline.
        X_train: The training feature dataset.
        y_train: The training target variable.
        X_test: The testing feature dataset.
        y_test: The testing target variable.

    Returns:
        None
    """
    # TODO add classifiers/regressors condition

    # Extract configuration settings from the pipeline_config dictionary
    dict_outliers = pipeline_config['features']['outliers_']
    dict_imputers = pipeline_config['features']['imputer_']
    estimator_dict = pipeline_config['estimator']
    metric_dict = pipeline_config['cv_technique']['metric']

    # Create a preprocessing pipeline
    outliers_handler_instance = Outliers(dict_outliers).run()
    impute_instance = Imputers(dict_imputers).run()

    pipeline_preprocess = Pipeline(
            [
                ('outliers_handler', outliers_handler_instance),
                ('imputation', impute_instance)
            ]
    )

    # Apply preprocessing to the training data
    x_train_eval = pipeline_preprocess.fit_transform(X_train)

    # Create and train the machine learning model
    model_instance = Classifiers(estimator_dict).run()
    model_instance.fit(x_train_eval, y_train)

    # Apply preprocessing to the test data
    X_test_eval = pipeline_preprocess.transform(X_test)

    # Make predictions on the test data
    y_pred = model_instance.predict(X_test_eval)
    y_true = y_test.values

    # Calculate and print the evaluation metric
    score = metric(metric_name=metric_dict['name'],
                   y_true=y_true,
                   y_pred=y_pred,
                   params=metric_dict['params'])

    print(f"Value ({metric_dict['name']}): {score}")
