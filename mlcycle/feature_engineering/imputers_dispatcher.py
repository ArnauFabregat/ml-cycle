from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import ArbitraryNumberImputer
from mlcycle.base import FeaturesBase
from sklearn.impute import KNNImputer
from typing import Callable, Dict, List


class Imputers(FeaturesBase):
    """
    A subclass of `FeaturesBase` representing a custom step for data imputation.

    Attributes:
        transformers_dict: A dictionary representing the configuration of the imputers.

    Examples:
        >>> imputers_dict = {
                1: {'features': ['mean radius', 'mean texture'],
                    'transformers': {
                        'KNNImputer': {'n_neighbors': 2}
                    },
                    'if_optuna': False
                },
                2: {'features': ['mean perimeter'],
                    'transformers': {
                        'ArbitraryNumberImputer': {'arbitrary_number': 5000}
                    },
                    'if_optuna': False
            }}
        >>> Imputers(imputers_dict)
    """

    def __init__(self, transformers_dict: Dict = {}) -> None:
        """
        Initializes an Imputers object.
        """
        super().__init__(transformers_dict, 'imputer')

    def _transformers_dispatcher(self, transformer_name: str, vars_list: List, params: Dict) -> Callable:
        """
        Defines specific imputers based on the transformer_name.

        Args:
            transformer_name: The name of the imputer transformer.
            vars_list: A list of variables to be imputed.
            params: The parameters for the imputer.

        Returns:
            Callable: The instantiated imputer transformer.

        Raises:
            ValueError: If the specified transformer_name is not supported.
        """
        if transformer_name == 'KNNImputer':
            transformer = SklearnTransformerWrapper(KNNImputer(
                **params
            ), variables=vars_list)

        elif transformer_name == 'ArbitraryNumberImputer':
            params['variables'] = vars_list
            transformer = ArbitraryNumberImputer(**params)

        else:
            raise ValueError(f"Unsupported transformer_name: {transformer_name}")

        return transformer
