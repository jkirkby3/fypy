from abc import ABC, abstractmethod
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.model.FourierModel import FourierModel
from typing import Union, Optional, Dict
import numpy as np
from contextlib import contextmanager


class LevyModel(FourierModel, ABC):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 params: np.ndarray,
                 frozen_params: Dict[float,list]= None):
        """
        Base class for an exponential Levy model, which is a model for which the characteristic function is known, hence
        enabling pricing by Fourier methods. These models are defined uniquely by their Levy "symbol", which determines
        the chf by:   chf(T,xi) = exp(T*symbol(xi)),  for all T>=0

        :param forwardCurve: Forward curve term structure
        """
        super().__init__(discountCurve=discountCurve,
                         forwardCurve=forwardCurve)
        self._forwardCurve = forwardCurve  # Overrides base forward
        self._params = params
        self.frozen_params = frozen_params if frozen_params is not None else {}

    @property
    def frozen_params(self):
        return self.__frozen_params

    @frozen_params.setter
    def frozen_params(self, frozen_params: Dict[float, list]):
        if not isinstance(frozen_params, dict):
            raise ValueError("frozen_params must be a dictionary")
        self.__frozen_params = frozen_params

    def chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """

        return np.exp(T * self.symbol(xi)) * self.frozen_chf(xi=xi) if self.frozen_params!=None else np.exp(T * self.symbol(xi))


    @abstractmethod
    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T * symbol(xi)), for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        raise NotImplementedError



    @abstractmethod
    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        raise NotImplementedError

    def risk_neutral_log_drift(self) -> float:
        """ Compute the risk-neutral drift of log process """
        return self.forwardCurve.drift(0, 1) + self.convexity_correction()

    def set_params(self, params: np.ndarray):
        self._params = params
        return self

    def get_params(self) -> np.ndarray:
        return self._params


    @contextmanager
    def temporary_params(self, new_params: np.ndarray):
        """
        This function temporarily sets the model's parameters (`_params`) to `new_params` for the duration of the
        context, and then automatically restores the original parameters after exiting the context.
        This method could be used for inhomogeneous Levy models, where the model's parameters must be changed frequently.

        :param new_params: np.ndarray, new set of parameters to use temporarily.
        """
        # Store the original parameters before modifying them
        original_params = self.get_params()
        try:
            # Set the new parameters
            self.set_params(new_params)
            yield
        finally:
            # Restore the original parameters after the context is done
            self.set_params(original_params)

    def frozen_chf(self, xi: float) -> complex:
        """
        :param xi: np.ndarray or float, points in the frequency domain
        :param thetas: list of np.ndarray, each array represents a set of parameters
        :param T: np.ndarray, array of time points T_j corresponding to each set of parameters.
        :return: np.ndarray or float, the frozen characteristic function.
        """

        frozen_factor = 1.0
        T_previous = 0

        for T, params in self.frozen_params.items():
            delta_T = T - T_previous
            T_previous = T

            with self.temporary_params(params):
                frozen_factor *= np.exp(delta_T * self.symbol(xi))

        return frozen_factor

    # def inhomogeneous_chf(self, T: float, xi: Union[float, np.ndarray], frozen_params: Dict[float, list] = None) -> complex:
    #     """
    #     Time-inhomogeneous characteristic function
    #     :param T: float, time to maturity
    #     :param xi: np.ndarray or float, points in frequency domain
    #     :param frozen_params: optional dict, parameters for the frozen characteristic function
    #     """
    #
    #     self._frozen_params = frozen_params or {}
    #
    #     return self.chf(T=T, xi=xi) * self.frozen_chf(xi=xi) if frozen_params else self.chf( T=T, xi=xi)
