from mlcycle.base import transformer_config, estimator_config
from mlcycle.feature_engineering.outliers_dispatcher import Outliers
from mlcycle.feature_engineering.imputers_dispatcher import Imputers
from mlcycle.modeling.cv_techniques_dispatcher import cross_validation_technique
from mlcycle.modeling.estimators_dispatcher import Classifiers
import optuna
from optuna.visualization import plot_optimization_history
import pandas
from sklearn.pipeline import Pipeline
from typing import Dict, Callable


class OptunaStudy():
    """
    The `OptunaStudy` class provides a way to optimize hyperparameters of a machine learning model
    using the Optuna library.

    Attributes:
        optuna_config: A dictionary containing configuration settings for the Optuna study.
        pipeline_config: A dictionary containing configuration settings for the final pipeline.
        X: The feature dataset.
        y: The target variable.
        study (optuna.study.Study): The Optuna study instance.
    """
    def __init__(self, config_yaml: Dict,
                 X: pandas.DataFrame,
                 y: pandas.DataFrame) -> None:
        """
        Initializes the `OptunaStudy` object.

        Args:
            config_yaml: A dictionary containing configuration settings for the Optuna study.
            X: The feature dataset.
            y: The target variable.
        """
        # TODO build a function to check the config_dict input format
        self.optuna_config = config_yaml.copy()
        self.pipeline_config = config_yaml.copy()
        self.X = X
        self.y = y
        self.study = self._study()

    def _study(self) -> Callable:
        """
        Creates and configures an Optuna study.

        Returns:
            Callable: The Optuna study instance.
        """
        params = self.optuna_config['optuna']['create_study']
        study = optuna.create_study(**params)
        return study

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Perform optimization by defining the space of hyperparameters and evaluating the performance
        using cross-validation.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object for hyperparameter optimization.

        Returns:
            float: The computed score result.
        """
        # Params config
        config = self.optuna_config

        # Extract configuration settings from the config dictionary
        dict_outliers = config['features']['outliers_']
        dict_imputers = config['features']['imputer_']
        estimator_dict = config['estimator']
        cv_technique_dict = config['cv_technique']

        # Create a preprocessing pipeline
        pipeline_preprocess = Pipeline(
            [
                ('outliers', Outliers(transformer_config(transformer_dict=dict_outliers,
                                                         if_optuna=True,
                                                         trial=trial)).run()),
                ('imputation', Imputers(transformer_config(transformer_dict=dict_imputers,
                                                           if_optuna=True,
                                                           trial=trial)).run())
            ]
        )

        # TODO add classifiers/regressors condition
        # Create and train the machine learning model
        pipeline_model = Classifiers(estimator_config(estimator_dict=estimator_dict,
                                                      if_optuna=True,
                                                      trial=trial)).run()

        # Apply preprocessing to the training data
        X_fe = pipeline_preprocess.fit_transform(self.X, self.y)

        # PIPELINE MODELO
        score_result = cross_validation_technique(cv_name=cv_technique_dict['cv_name'],
                                                  params=cv_technique_dict['params'],
                                                  X=X_fe,
                                                  y=self.y,
                                                  model=pipeline_model,
                                                  metric_dict=cv_technique_dict['metric'])
        return score_result

    def run(self, n_trials: int = 10) -> None:
        """
        This method runs the Optuna optimization process.

        Args:
            n_trials (int): The number of optimization trials to perform.

        Returns:
            None
        """
        # Params config
        config = self.optuna_config

        # Activate study
        self.study.optimize(self._objective, n_trials=n_trials)

        # Optimized config
        # del TRIAL / OPTUNA into False
        for k1 in list(config['features'].keys()):
            for k in list(config['features'][k1].keys()):
                config['features'][k1][k]['if_optuna'] = False
                del config['features'][k1][k]['trial']

        del config['estimator']['trial']
        config['estimator']['if_optuna'] = False

        # Keep just winner transformers
        optuna_params = self.study.best_params

        for k1 in list(config['features'].keys()):
            for i in [k for k in optuna_params.keys() if k1 in k]:
                num = int(i.split('_')[-1])
                ks = list(config['features'][k1][num]['transformers'].keys())
                for k in ks:
                    if k != optuna_params[i]:
                        del config['features'][k1][num]['transformers'][k]

        self.pipeline_config = config

    def results(self, verbose: bool = True, plot: bool = True) -> None:
        """
        Displays the results of the Optuna optimization.

        Args:
            verbose (bool): If True, print optimization results.
            plot (bool): If True, display an optimization history plot.

        Returns:
            None
        """
        if verbose:
            metric_name = self.pipeline_config['cv_technique']['metric']['name']
            print(f"\tBest value ({metric_name}): {self.study.best_value:.5f}")
            print(f"\tBest params:")
            for key, value in self.study.best_params.items():
                print(f"\t\t{key}: {value}")

        if plot:
            fig = plot_optimization_history(self.study)
            fig.show()
