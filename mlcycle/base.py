import abc
import optuna
from sklearn.pipeline import Pipeline
from typing import Dict, Callable, List


class FeaturesBase():
    """
    The `FeaturesBase` class provides a framework for defining custom data
    transformation steps using scikit-learn pipelines. Subclasses can inherit
    from this class and implement the `_transformers_dispatcher` method to
    specify the actual transformers to be used in the pipeline.

    Attributes:
        name: The name of the custom step.
        transformers_list: A dictionary representing the configuration of the
            transformers used in the step.
    """
    def __init__(self, transformers_dict: Dict = {}, name: str = None) -> None:
        """
        Initializes the `CustomStep` object.
        """
        # TODO build a function to check the config_dict input format
        self.name = name
        self.config_dict = transformers_dict

    def run(self) -> Callable:
        """
        This method runs the class main function.

        Returns:
            Output of the `_create_pipeline()` method.
        """
        return self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """
        This method generates a scikit-learn pipeline for feature transformations. The method
        iterates over the configuration dictionary and adds each transformer to the pipeline.

        Returns:
            A *scikit-learn* `Pipeline` object representing the transformations pipeline.
        """
        # Pipeline initialization
        pipeline_instance = Pipeline(
            [
                (self.name, None),
            ],
        )
        # Transformers to pipeline
        for count, step in self.config_dict.items():
            # Extract parameters from dict
            transformers_list = list(step['transformers'].keys())
            vars_list = step['features']
            if_optuna = step['if_optuna']

            # Transformer name selector
            if if_optuna:
                trial = step['trial']
                transformer_name = trial.suggest_categorical(
                    name=f"{self.name}_{count}",
                    choices=transformers_list
                )
            else:
                transformer_name = transformers_list[0]

            # Transformer parameters definition
            params = step['transformers'][transformer_name]

            # Update params dictionary with optuna trial parameters
            if if_optuna:
                params = self._trial_params(params=params,
                                            trial=trial,
                                            transformer_name=transformer_name,
                                            count=count)

            # Define specific transformers
            transformer = self._transformers_dispatcher(transformer_name=transformer_name,
                                                        vars_list=vars_list,
                                                        params=params)

            # Adding step to pipeline
            pipeline_instance = Pipeline(
                steps=pipeline_instance.steps +
                [
                    (f'{transformer_name}_{count}', transformer),
                ],
            )
        # Remove the initialization step
        if len(pipeline_instance.steps) > 1:
            pipeline_instance.steps.pop(0)
        return pipeline_instance

    def _trial_params(self, params: Dict, trial: optuna.trial.Trial, transformer_name: str, count: int) -> Dict:
        """
        This method is a helper function that updates the transformer parameters with optuna trial parameters.

        Args:
            params: The original parameters of the transformer.
            trial: The optuna trial object used for parameter optimization.
            transformer_name: The name of the transformer.
            count: The count representing the order of the transformer in the pipeline.

        Returns:
            The updated parameters dictionary with trial parameters.

        """
        for name, values in params.items():
            # Trial params initialization
            trial_param = values

            if type(values) == tuple:
                low = values[0]
                high = values[1]
                trial_step = values[2]

                if type(low) == float or type(high) == float or type(trial_step) == float:
                    trial_param = trial.suggest_float(name=f"{transformer_name}_{name}_{count}",
                                                      low=low,
                                                      high=high,
                                                      step=trial_step)
                else:
                    trial_param = trial.suggest_int(name=f"{transformer_name}_{name}_{count}",
                                                    low=low,
                                                    high=high,
                                                    step=trial_step)

            if type(values) == list:
                trial_param = trial.suggest_categorical(name=f"{transformer_name}_{name}_{count}",
                                                        choices=values)

            # Update params dictionary with trial parameters
            params[name] = trial_param

        return params

    @abc.abstractmethod
    def _transformers_dispatcher(self, transformer_name: str, vars_list: List, params: Dict) -> Callable:
        """
        This is an abstract method that needs to be implemented by subclasses.
        It serves as a placeholder for adding the transformer dispatcher.

        Defines specific transformers based on the transformer_name.

        Args:
            transformer_name: The name of the transformer.
            vars_list: A list of variables to be transformed.
            params: The parameters for the transformer.

        Returns:
            The instantiated transformer.

        """
        return Callable


class EstimatorsBase():
    """
    The `EstimatorsBase` class provides a framework for defining
    estimators using scikit-learn pipelines. Subclasses can inherit
    from this class and implement the `_models_dispatcher` method to
    specify the actual models to be used in the pipeline.

    Attributes:
        estimator_dict: A dictionary representing the configuration of the
            estimator used in the step.
    """
    def __init__(self, estimator_dict: Dict = {}) -> None:
        """
        Initializes the `EstimatorsBase` object.
        """
        # TODO build a function to check the config_dict input format
        self.config_dict = estimator_dict

    def run(self) -> Callable:
        """
        This method runs the class main function.

        Returns:
            Output of the `_create_pipeline()` method.
        """
        return self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """
        This method generates a scikit-learn pipeline for estimators. The method
        iterates over the configuration dictionary and adds the estimator to the pipeline.

        Returns:
            A *scikit-learn* `Pipeline` object representing the transformations pipeline.
        """

        # Extract parameters from dict
        estimator_name = self.config_dict['estimator_name']
        params = self.config_dict['params']
        if_optuna = self.config_dict['if_optuna']
        if if_optuna:
            trial = self.config_dict['trial']

        # Update params dictionary with optuna trial parameters
        if if_optuna:
            params = self._trial_params(params=params,
                                        trial=trial,
                                        estimator_name=estimator_name)

        # Define specific estimator
        estimator = self._estimators_dispatcher(estimator_name=estimator_name,
                                                params=params)

        # Adding step to pipeline
        pipeline_instance = Pipeline(
            [
                (f'{estimator_name}', estimator),
            ]
        )

        return pipeline_instance

    def _trial_params(self, params: Dict, trial: optuna.trial.Trial, estimator_name: str) -> Dict:
        """
        This method is a helper function that updates the estimator parameters with optuna trial parameters.

        Args:
            params: The original parameters of the estimator.
            trial: The optuna trial object used for parameter optimization.
            estimator_name: The name of the estimator.

        Returns:
            The updated parameters dictionary with trial parameters.

        """
        for name, values in params.items():
            # Trial params initialization
            trial_param = values

            if type(values) == tuple:
                low = values[0]
                high = values[1]
                trial_step = values[2]

                if type(low) == float or type(high) == float or type(trial_step) == float:
                    trial_param = trial.suggest_float(name=f"{estimator_name}_{name}",
                                                      low=low,
                                                      high=high,
                                                      step=trial_step)
                else:
                    trial_param = trial.suggest_int(name=f"{estimator_name}_{name}",
                                                    low=low,
                                                    high=high,
                                                    step=trial_step)

            if type(values) == list:
                trial_param = trial.suggest_categorical(name=f"{estimator_name}_{name}",
                                                        choices=values)

            # Update params dictionary with trial parameters
            params[name] = trial_param

        return params

    @abc.abstractmethod
    def _estimators_dispatcher(self, estimator_name: str, params: Dict) -> Callable:
        """
        This is an abstract method that needs to be implemented by subclasses.
        It serves as a placeholder for adding the estimators dispatcher.

        Defines specific estimators based on the estimator_name.

        Args:
            estimator_name: The name of the estimator.
            params: The parameters for the estimator.

        Returns:
            The instantiated estimator.
        """
        return Callable


def transformer_config(transformer_dict: dict,  if_optuna: bool = False, trial: optuna.trial.Trial = None):
    """
    Configures a dictionary representing a transformer with optional Optuna integration.

    Args:
        transformer_dict (dict): A dictionary representing the configuration of a transformer.
        if_optuna (bool, optional): If True, integrates Optuna hyperparameter optimization by
                                    adding a 'trial' key to the dictionary. Defaults to False.
        trial (optuna.trial.Trial, optional): An Optuna Trial object used for hyperparameter optimization.
                                              Required if if_optuna is True.

    Returns:
        dict: The modified transformer_dict with an optional 'trial' key added for Optuna integration.

    Example:
        transformer_dict = {'layers': 4, 'dropout': 0.2}
        transformed_dict = transformer_config(transformer_dict, if_optuna=True, trial=my_trial)
    """
    if if_optuna:
        # Integrate Optuna by adding the 'trial' object to each sub-dictionary in transformer_dict.
        for k in transformer_dict.keys():
            transformer_dict[k]['trial'] = trial

    return transformer_dict


def estimator_config(estimator_dict: dict,  if_optuna: bool = False, trial: optuna.trial.Trial = None):
    """
    Configures a dictionary representing an estimator with optional Optuna integration.

    Args:
        estimator_dict (dict): A dictionary representing the configuration of an estimator.
        if_optuna (bool, optional): If True, integrates Optuna hyperparameter optimization by
                                    adding a 'trial' key to the dictionary. Defaults to False.
        trial (optuna.trial.Trial, optional): An Optuna Trial object used for hyperparameter optimization.
                                              Required if if_optuna is True.

    Returns:
        dict: The modified estimator_dict with an optional 'trial' key added for Optuna integration.

    Example:
        estimator_dict = {'model': 'RandomForest', 'n_estimators': 100}
        configured_estimator = estimator_config(estimator_dict, if_optuna=True, trial=my_trial)
    """
    if if_optuna:
        # Integrate Optuna by adding the 'trial' object to estimator_dict.
        estimator_dict['trial'] = trial

    return estimator_dict
