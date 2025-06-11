from feature_engine.outliers import Winsorizer
from feature_engine.transformation import LogTransformer
from feature_engine.transformation import PowerTransformer
from mlcycle.base import FeaturesBase
from typing import Callable, Dict, List, Optional
import pandas
import numpy
from sklearn.base import BaseEstimator, TransformerMixin


class Outliers(FeaturesBase):
    """
    A subclass of `FeaturesBase` representing a custom step for outliers
    treatment.

    Attributes:
        outliers_dict: A dictionary representing the configuration of the
            outliers techniques.

    Examples:
        >>> outliers_dict = {
            1: {'features': ['mean radius'],
                'transformers': {
                'Winsorizer': {'capping_method': 'gaussian',
                                'tail': 'both',
                                'fold': 3}
                },
                'if_optuna': False
            },
            2: {'features': ['mean perimeter'],
                'transformers': {'LogTransformer': {'base': 'e'}
                },
                'if_optuna': False
            },
            3: {'features': ['mean texture'],
                'transformers': {'PowerTransformer': {'exp': 0.5}
                },
                'if_optuna': False
            }
        >>> outliers(outliers_dict)
    """

    def __init__(self, transformers_dict: Dict = {}) -> None:
        """
        Initializes an Outliers object.
        """
        super().__init__(transformers_dict, 'outliers')

    def _transformers_dispatcher(self, transformer_name: str, vars_list: List, params: Dict) -> Callable:
        """
        Defines specific outlier transformers based on the transformer_name.

        Args:
            transformer_name: The name of the outlier transformer.
            vars_list: A list of variables to be imputed.
            params: The parameters for the outlier transformer.

        Returns:
            Callable: The instantiated outlier transformer.

        Raises:
            ValueError: If the specified transformer_name is not supported.
        """

        if transformer_name == 'LogTransformer':
            params['variables'] = vars_list
            transformer = LogTransformer(**params)

        elif transformer_name == 'PowerTransformer':
            params['variables'] = vars_list
            transformer = PowerTransformer(**params)

        elif transformer_name == 'CustomWinsorizer':
            params['variables'] = vars_list
            transformer = CustomWinsorizer(params)

        else:
            raise ValueError(f"Unsupported transformer_name: {transformer_name}")

        return transformer


class CustomWinsorizer(BaseEstimator, TransformerMixin):
    """
    Custom version of the Winsorizer Outlier Handler from Feature Engine.
    The Winsorizer() caps maximum and/or minimum values of a variable at
    automatically determined values, and optionally adds indicators.
    The Custom Version adds the option of setting these outliers to nan.

    Attributes:
        params: A dictionary representing the configuration of the
            CustomWinsorizer. default={'capping_method': 'gaussian',
                                       'tail': 'right',
                                       'fold': 3,
                                       'add_indicators': False,
                                       'variables': None,
                                       'missing_values': 'ignore',
                                       'outliers_to_nan': False}
        outliers_to_nan: A boolean indicating to set outliers to NaN or not.

    Examples:
        >>> outliers_dict = {
            1: {'features': ['mean radius'],
                'transformers': {
                'CustomWinsorizer': {'capping_method': 'gaussian',
                                     'tail': 'both',
                                     'fold': 3,
                                     'add_indicators': True,
                                     'outliers_to_nan': True}
                },
                'if_optuna': False
                }
            }
        >>> output = outliers(outliers_dict).fit_transform(X)
    """
    def __init__(self, params: Dict = {}) -> None:
        """
        Initializes the `CustomWinsorizer` object.
        """
        self.params = params
        if 'missing_values' not in self.params:
            self.params['missing_values'] = 'ignore'
        if 'outliers_to_nan' not in self.params:
            self._outliers_to_nan = False
        else:
            self._outliers_to_nan = params['outliers_to_nan']
        self._variables = params['variables']
        self._params_Winsorizer = self._set_params_Winsorizer()
        self._transformer = Winsorizer(**self._params_Winsorizer)

    def fit(self, X: pandas.DataFrame, y: Optional[pandas.Series] = None) -> Callable:
        """
        Identify outliers and learn the values to replace them with.

        Args:
            X: Pandas DataFrame with the training data.
            y: Pandas Series, default=None not needed for the transformed.

        Returns:
            self: The CustomWinsorizer transformer fitted.
        """
        if self._outliers_to_nan is False:

            self._transformer.fit(X)
        else:
            self._params_Winsorizer['add_indicators'] = True
            self._transformer.fit(X)
        return self

    def transform(self, X: pandas.DataFrame) -> pandas.DataFrame:
        """
        Cap the variable values. Optionally, add outlier indicators and/or
        set outliers to numpy.NaN. If outliers_to_nan == True it will
        set the outliers to numpy.NaN and if add_indicators=False then it will
        delete them.

        Args:
            X: Pandas DataFrame of shape = [n_samples, n_features]
                with the training data.

        Returns:
            X_: pandas dataframe of shape = [n_samples, n_features + n_ind]
                The dataframe containing the capped variables and indicators.
                The number of output variables depends on the values for 'tail'
                and 'add_indicators': add_indicators=False is passed, it will
                be equal to 'n_features'; otherwise, it will have an additional
                indicator column for each processed feature for both tails.
        """
        X_ = self._transformer.transform(X)
        if self._outliers_to_nan is True:
            X_ = self._to_nan(X_)
        if 'add_indicators' not in self.params or self.params['add_indicators'] is False:
            X_ = self._delete_indicators(X_)
        return X_

    def _to_nan(self, X_: pandas.DataFrame) -> pandas.DataFrame:
        """
        Set outliers to numpy.NaN.

        Args:
            X_: Pandas DataFrame of shape = [n_samples, n_features + n_ind]
                with the training data already transformed and the indicators added.
        Returns:
            X_: pandas dataframe of shape = [n_samples, n_features + n_ind]
                The dataframe containing the capped variables and indicators
                and the outliers set to numpy.NaN.
        """
        if self._transformer.tail in ['right', 'both']:
            indicators = [str(cl) + "_right" for cl in self._variables]
            for col, var in zip(indicators, self._variables):
                index = X_[X_[col] == 1].index
                X_.loc[index, var] = numpy.NaN

        if self._transformer.tail in ['left', 'both']:
            indicators = [str(cl) + "_left" for cl in self._variables]
            for col, var in zip(indicators, self._variables):
                index = X_[X_[col] == 1].index
                X_.loc[index, var] = numpy.NaN
        return X_

    def _delete_indicators(self, X_: pandas.DataFrame) -> pandas.DataFrame:
        """
        Delete indicators columns.

        Args:
            X_: Pandas DataFrame of shape = [n_samples, n_features + n_ind]
                with the training data already transformed and the indicators added.
        Returns:
            X_: pandas dataframe of shape = [n_samples, n_features]
                The dataframe containing the capped variables with the additional
                indicator columns for each processed feature for both tails
                eliminated.
        """
        indicators = []
        if self._transformer.tail in ['right', 'both']:
            indicators += [str(cl) + "_right" for cl in self._variables]
        if self._transformer.tail in ['left', 'both']:
            indicators += [str(cl) + "_left" for cl in self._variables]
        X_ = X_.drop(indicators, axis=1)
        return X_

    def _set_params_Winsorizer(self) -> Dict:
        """
        Creates a dictionary with the correct parameters to use in the
        Winsorizer object from Feature Engine from self.params atribute.

        Returns:
            Dictionary with the parameters to  Winsorizer Outlier Handler.
        """
        params_Winsorizer = self.params.copy()
        if 'outliers_to_nan' in self.params:
            del params_Winsorizer['outliers_to_nan']
        params_Winsorizer['add_indicators'] = True
        return params_Winsorizer
