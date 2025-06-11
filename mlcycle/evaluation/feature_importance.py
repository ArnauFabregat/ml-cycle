from matplotlib import pyplot
import pandas
import shap
from sklearn.base import BaseEstimator


class shapValues():
    """ Class to generate several shap values plots.

    Args:
        model: The machine learning model to be explained using SHAP.
        data: The dataframe containing the input data to the model.
    """

    def __init__(self, model: BaseEstimator, data: pandas.DataFrame):

        self.model = model
        self.data = data
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(self.data)

    def summaryPlot(self, class_idx: int) -> None:
        """
        Generates a summary plot of SHAP values for the given model and dataframe.
        
        Parameters:
            class_idx: The index of the class to be explained (0 or 1 in case of binary classification).

        Returns:
            Plot.
        """

        shap.summary_plot(self.shap_values[class_idx], self.data, plot_type='bar')

    def beeswarmPlot(self, class_idx: int) -> None:
        """
        Generates a beeswarm plot of SHAP values for the given model and dataframe.
        
        Parameters:
            class_idx: The index of the class to be explained (0 or 1 in case of binary classification).

        Returns:
            Plot.
        """
        shap.summary_plot(self.shap_values[class_idx], self.data, show=False, color_bar=False)
        pyplot.colorbar()
        pyplot.show()

    def shapPlot(self, row_idx: int, class_idx: int) -> None:
        """
        Generates a force plot of SHAP values for a given row of the input dataframe.

        Parameters:
            row_idx: The index of the row to be explained.
            class_idx: The index of the class to be explained (0 or 1 in case of binary classification).

        Returns:
            A force plot of SHAP values for the given row.
        """
        shap.initjs()
        shap_values = self.explainer.shap_values(self.data.iloc[row_idx])
        p = shap.force_plot(self.explainer.expected_value[class_idx], shap_values[class_idx], self.data.iloc[row_idx])
        return p
