from lightgbm import LGBMClassifier, LGBMRegressor
from mlcycle.base import EstimatorsBase
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Callable, Dict


class Classifiers(EstimatorsBase):
    """
    A subclass of `EstimatorsBase` representing a collection of classifiers.

    Attributes:
        estimator_dict: A dictionary representing the configuration of the estimator.

    Examples:
        >>> estimator_dict = {
                'estimator_name': 'RandomForestClassifier',
                'params': {'n_estimators': 50},
                'if_optuna': False
            }
        >>> Classifiers(estimator_dict)
    """

    def __init__(self, estimator_dict: Dict = {}) -> None:
        """
        Initializes a Classifiers object.
        """
        super().__init__(estimator_dict)

    def _estimators_dispatcher(self, estimator_name: str, params: Dict) -> Callable:
        """
        Defines specific imputers based on the estimator_name.

        Args:
            estimator_name: The name of the estimator.
            params: The parameters for the estimator.

        Returns:
            The instantiated estimator.

        Raises:
            ValueError: If the specified estimator_name is not supported.
        """
        # Linear models
        if estimator_name == 'LogisticRegression':
            model = LogisticRegression(**params)

        # Tree-based models
        elif estimator_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**params)

        elif estimator_name == 'LGBMClassifier':
            model = LGBMClassifier(**params)

        else:
            raise ValueError(f"Unsupported estimator_name: {estimator_name}")

        return model


class Regressors(EstimatorsBase):
    """
    A subclass of `EstimatorsBase` representing a collection of regressors.

    Attributes:
        estimator_dict: A dictionary representing the configuration of the estimator.

    Examples:
        >>> estimator_dict = {
                'estimator_name': 'RandomForestRegressors',
                'params': {'n_estimators': 50},
                'if_optuna': False
            }
        >>> Regressors(estimator_dict)
    """

    def __init__(self, estimator_dict: Dict = {}) -> None:
        """
        Initializes a Regressors object.
        """
        super().__init__(estimator_dict)

    def _estimators_dispatcher(self, estimator_name: str, params: Dict) -> Callable:
        """
        Defines specific imputers based on the estimator_name.

        Args:
            estimator_name: The name of the estimator.
            params: The parameters for the estimator.

        Returns:
            The instantiated estimator.

        Raises:
            ValueError: If the specified estimator_name is not supported.
        """
        # Linear models
        if estimator_name == 'LogisticRegression':
            model = LinearRegression(**params)

        # Tree-based models
        elif estimator_name == 'RandomForestClassifier':
            model = RandomForestRegressor(**params)

        elif estimator_name == 'LGBMClassifier':
            model = LGBMRegressor(**params)

        else:
            raise ValueError(f"Unsupported estimator_name: {estimator_name}")

        return model
