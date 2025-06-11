import matplotlib.pyplot
import numpy
import pandas
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


class Monotonic_calibrator():
    """
    An isotonic calibrator model class.

    Attributes:
        calibrator (sklearn IsotonicRegression object): Calibrator model to calibrate the output requested.
    """

    def __init__(self) -> None:
        """This method create and initialize the non-decreasing calibrator.
        """
        self.calibrator = IsotonicRegression()

    def fit(self,
            no_calibrated_proba_tr: numpy.ndarray,
            y_tr: numpy.ndarray,
            no_calibrated_proba_ts: numpy.ndarray = numpy.zeros(0),
            y_ts: numpy.ndarray = numpy.zeros(0),
            plot_curves: bool = False) -> None:
        """This method fit the isotonic calibrator.

        Args:
            no_calibrated_proba_tr: The output from the probability model for the N trian samples.
            y_tr: The training target vector Nx1 containing the label for N samples.
            no_calibrated_proba_ts:  The output from the probability model for the N trian samples.
            y_ts: The testing target vector Nx1 containing the label for N samples.
            plot_curves: If True, calibration curves will be plotted.
        """
        if type(no_calibrated_proba_tr) in [pandas.core.frame.DataFrame, pandas.core.series.Series]:
            no_calibrated_proba_tr = no_calibrated_proba_tr.values
        if type(y_tr) in [pandas.core.frame.DataFrame, pandas.core.series.Series]:
            y_tr = y_tr.values
        if type(no_calibrated_proba_ts) in [pandas.core.frame.DataFrame, pandas.core.series.Series]:
            no_calibrated_proba_ts = no_calibrated_proba_ts.values
        if type(y_ts) in [pandas.core.frame.DataFrame, pandas.core.series.Series]:
            y_ts = y_ts.values

        self.calibrator.fit(no_calibrated_proba_tr, y_tr)

        if plot_curves:

            print("Train set curves")
            calibrated_proba_tr = self.calibrate_proba(no_calibrated_proba=no_calibrated_proba_tr)
            calibrated_proba_ts = self.calibrate_proba(no_calibrated_proba=no_calibrated_proba_ts)

            self.show_calibration_curves(no_calibrated_proba_tr=no_calibrated_proba_tr,
                                         calibrated_proba_tr=calibrated_proba_tr,
                                         y_true_tr=y_tr,
                                         no_calibrated_proba_ts=no_calibrated_proba_ts,
                                         calibrated_proba_ts=calibrated_proba_ts,
                                         y_true_ts=y_ts
                                         )

    def calibrate_proba(self, no_calibrated_proba: numpy.ndarray) -> numpy.ndarray:
        """This method predict the calibrated probability for each sample.

        Args:
            no_calibrated_proba:  The output from the probability model for the N samples.

        Returns:
            calibrated_proba: The calibrated probability.
        """
        if type(no_calibrated_proba) in [pandas.core.frame.DataFrame, pandas.core.series.Series]:
            no_calibrated_proba = no_calibrated_proba.values

        calibrated_proba = self.calibrator.predict(no_calibrated_proba)
        return calibrated_proba

    def show_calibration_curves(self,
                                no_calibrated_proba_tr: numpy.ndarray,
                                calibrated_proba_tr: numpy.ndarray,
                                y_true_tr: numpy.ndarray,
                                no_calibrated_proba_ts: numpy.ndarray = numpy.zeros(0),
                                calibrated_proba_ts: numpy.ndarray = numpy.zeros(0),
                                y_true_ts: numpy.ndarray = numpy.zeros(0)) -> None:
        """This method show the calibration curves of the base estimator and the calibrator model for the received
           input.

        Args:
            no_calibrated_proba_tr: The output from the probability model for the N samples of the train set.
            calibrated_proba_tr: The output from the calibration model for the N samples of the train set.
            y_true_tr: The target values of the train set (class labels in classification, real numbers in regression).
            no_calibrated_proba_ts: The output from the probability model for the N samples of the test set.
            calibrated_proba_ts: The output from the calibration model for the N samples of the test set.
            y_true_ts: The target values of the test set (class labels in classification, real numbers in regression).
        """

        fraccion_positivos_tr, media_prob_predicha_tr = calibration_curve(list(y_true_tr),
                                                                          list(no_calibrated_proba_tr),
                                                                          n_bins=20)
        fraccion_positivos_ts, media_prob_predicha_ts = calibration_curve(list(y_true_ts),
                                                                          list(no_calibrated_proba_ts),
                                                                          n_bins=20)

        fig, axs = matplotlib.pyplot.subplots(nrows=2, ncols=3, figsize=(6*3, 2*3.84))

        axs[0, 0].plot(media_prob_predicha_tr, fraccion_positivos_tr, "s-", color="b", label="Modelo base tr")
        axs[0, 0].plot(media_prob_predicha_ts, fraccion_positivos_ts, "s-", color="r", label="Modelo base ts")
        axs[0, 0].plot([0, 1], [0, 1], "k:", label="Calibración perfecta")
        axs[0, 0].set_ylabel("Proporción de clasificación correcta")
        axs[0, 0].set_xlabel("Probabilidad media estimada por el modelo (predict_proba)")
        axs[0, 0].set_title('Curva de calibrado (reliability curve)')
        axs[0, 0].legend()

        axs[1, 0].hist(no_calibrated_proba_tr,
                       range=(0, 1),
                       bins=10,
                       density=True,
                       lw=2,
                       alpha=0.3,
                       color="b",
                       label="tr")
        axs[1, 0].hist(no_calibrated_proba_ts,
                       range=(0, 1),
                       bins=10,
                       density=True,
                       lw=2,
                       alpha=0.3,
                       color="r",
                       label="ts")
        axs[1, 0].set_xlabel("Probabilidad estimada por el modelo (predict_proba)")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].set_title('Distribución de las probabilidades predichas por el modelo')
        axs[1, 0].legend()

        fraccion_positivos_tr, media_prob_predicha_tr = calibration_curve(list(y_true_tr),
                                                                          list(calibrated_proba_tr),
                                                                          n_bins=20)
        fraccion_positivos_ts, media_prob_predicha_ts = calibration_curve(list(y_true_ts),
                                                                          list(calibrated_proba_ts),
                                                                          n_bins=20)

        axs[0, 1].plot(media_prob_predicha_tr, fraccion_positivos_tr, "s-", color="b", label="Calibrador tr")
        axs[0, 1].plot(media_prob_predicha_ts, fraccion_positivos_ts, "s-", color="r", label="Calibrador ts")
        axs[0, 1].plot([0, 1], [0, 1], "k:", label="Calibración perfecta")
        axs[0, 1].set_ylabel("Proporción de clasificación correcta")
        axs[0, 1].set_xlabel("Probabilidad media estimada por el calibrador (predict_proba)")
        axs[0, 1].set_title('Curva de calibrado (reliability curve)')
        axs[0, 1].legend()

        axs[1, 1].hist(calibrated_proba_tr,
                       range=(0, 1),
                       bins=10,
                       density=True,
                       lw=2,
                       alpha=0.3,
                       color="b",
                       label="tr")
        axs[1, 1].hist(calibrated_proba_ts,
                       range=(0, 1),
                       bins=10,
                       density=True,
                       lw=2,
                       alpha=0.3,
                       color="r",
                       label="ts")
        axs[1, 1].set_xlabel("Probabilidad estimada por el calibrador (predict_proba)")
        axs[1, 1].set_ylabel("Count")
        axs[1, 1].legend()
        axs[1, 1].set_title('Distribución de las probabilidades predichas por el calibrador')

        axs[0, 2].plot(no_calibrated_proba_tr, calibrated_proba_tr, 'b.', label="tr")
        axs[0, 2].plot(no_calibrated_proba_ts, calibrated_proba_ts, 'r.', label="ts")
        axs[0, 2].set_xlabel("No calibrated probs")
        axs[0, 2].set_ylabel("Calibrated probs")
        axs[0, 2].set_title('Desplazamiento de las probabilidades')
        axs[0, 2].legend()

        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
