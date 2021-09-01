from fypy.fit.Calibratable import Calibratable
from fypy.fit.Minimizer import Minimizer, OptResult
from fypy.fit.Objective import Objective
from typing import Dict, Union, List, Tuple, Optional
import numpy as np


class Calibrator(object):
    def __init__(self,
                 model: Calibratable,
                 minimizer: Minimizer):
        """
        A generic calibration engine. The idea is to supply a minimizer of some kind (least squares), as well
        as a calibratable object of some kind, e.g. a full model or a component of the model, and to calibrate
        that model based on the set of objectives/penalties that are added to this calibrator. At the least, you
        should add some set of targets. e.g., to calibrate a stochastic volatility (SV) model to a set of market prices,
        add a Targets objective, containing the market prices, and supply the SV model to calibrate. Upon completion
        of the calibration, the model parameters will be set to their calibrated values, and your model is ready
        for pricing/risk.
        :param model: Calibratable, some calibratable object/model (e.g. Levy model)
        :param minimizer: Minimizer, some minimizer (e.g Levenberg-Marquardt least squares)
        """
        self._model = model
        self._minimizer = minimizer

        self._objectives: Dict[str, object] = {}
        self._guess: Optional[np.ndarray] = None
        self._bounds: Union[Tuple, List[Tuple]] = []

    def add_objective(self,
                      name: str,
                      objective: Objective):
        """
        Add a new objective function to the set of objectives. You can calibrate with as many objectives as you want,
        some representing targets, others representing regularization penalties, aribtrage penalties, etc.
        Each objective is named. Adding an objective with an existing name overwrites that objective
        :param name: str, the name of this objective
        :param objective: Objective, an objective that will guide the calibration process
        :return: self
        """
        self._objectives[name] = objective
        return self

    def set_bounds(self, bounds: Union[Tuple, List[Tuple]]):
        """
        Set the bounds on parameters
        :param bounds: the bounds per parameter, list of (lower,upper) bounds per parameter
        :return: self
        """
        self._bounds = bounds
        return self

    def set_initial_guess(self, params: np.ndarray):
        """
        Set the initial guess used to start the calibration
        :param params: np.ndarray, initial guess for parameters
        :return: self
        """
        self._guess = params
        return self

    def calibrate(self) -> OptResult:
        """ Run the calibration, fits the model in place, returns the optimization summary """
        return self._minimizer.minimize(self._objective_value,
                                        bounds=self._bounds, guess=self._guess)

    # ========================
    # Private
    # ========================

    def _objective_value(self, params: np.ndarray) -> np.ndarray:
        if len(self._objectives) == 0:
            raise RuntimeError("You never set any objectives ")

        self._model.set_params(params)
        return np.concatenate(objective.value() for _, objective in self._objectives)
