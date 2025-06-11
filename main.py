"""
Run:
    python main.py config/base_config.yml
"""
import sys
import yaml
from typing import Dict

# from util.logger import logger
from mlcycle.modeling.hyperparameter_tunning import OptunaStudy
from mlcycle.modeling.serving import fit_model, train_test_eval

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main(config_dict: Dict) -> None:
    """
    This function acts as an orchestrator of the process of ...

    Args:
        config_dict (dict): a dictionary that contains the information needed for the correct execution
                            of the script.
    Returns:
        None
    """
    # TODO Data - load
    data = load_breast_cancer(as_frame=True)
    # logger.info('Loaded input data')

    # TODO Data - train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data['data'], data['target'].map({0: 1, 1: 0}), test_size=0.33, random_state=42)

    # Train - hyperparameter tunning
    optuna_study = OptunaStudy(config_dict, X_train, y_train)
    optuna_study.run(n_trials=10)
    # logger.info('Parameter optimization done')

    # TODO Train - training html report

    # Test
    train_test_eval(pipeline_config=optuna_study.pipeline_config,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)

    # TODO Test - metrics html report

    # Serving
    fit_model(pipeline_config=optuna_study.pipeline_config,
              X=data['data'],
              y=data['target'].map({0: 1, 1: 0}),
              save=True)

    # logger.info('Process finished!')


if __name__ == "__main__":

    # Load configuration yaml
    config_path = sys.argv[1]
    with open(config_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # logger.info('Loaded configuration')
    main(config_dict)
