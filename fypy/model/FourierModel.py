from abc import ABC, abstractmethod
from fypy.fit.Calibratable import Calibratable
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.termstructures.ForwardCurve import ForwardCurve
from typing import Union
import numpy as np


class Cumulants(object):
    def __init__(self,
                 T: float,
                 rn_drift: float = np.nan,
                 c1: float = np.nan,
                 c2: float = np.nan,
                 c4: float = np.nan):
        """
        Cumulants and drift of the log process at a particular point in time
        :param T: float, time corresponding to cumulants
        :param rn_drift: float, (risk-neutral) drift of log process
        :param c1: float, first cumulant
        :param c2: float, second cumulant
        :param c4: float, fourth cumulant
        """
        self.T = T
        self.rn_drift = rn_drift
        self.c1 = c1
        self.c2 = c2
        self.c4 = c4

    def get_truncation_heuristic(self, L: float = 10.) -> float:
        """
        Calculates density truncation Heuristic of Fang and Oosterlee 2008
        :param L: float, increase this to increase gridwidth
        :return: float, a heuristic truncation width based on cumulants
        """
        return L * np.sqrt(abs(self.c2) + np.sqrt(abs(self.c4)))


class FourierModel(Calibratable, ABC):
    def __init__(self,
                 discountCurve: DiscountCurve,
                 forwardCurve: ForwardCurve):
        """
        Base class for a "Fourier" model, which is a model for which the characteristic function is known, hence
        enabling pricing by Fourier methods

        :param discountCurve: Discount curve term structure
        :param forwardCurve: Forward curve term structure
        """
        self._discountCurve = discountCurve
        self._forwardCurve = forwardCurve

    def spot(self) -> float:
        """
        Get the spot. In the case where a spot doesn't really make sense, it can simply return the default
        implementation, which is forward at time = 0
        """
        return self._forwardCurve.spot()

    @property
    def discountCurve(self) -> DiscountCurve:
        """ Get the discount Curve term structure """
        return self._discountCurve

    @property
    def forwardCurve(self) -> ForwardCurve:
        """ Get the forwarding term structure """
        return self._forwardCurve

    # ============================
    # Fourier interface
    # ============================

    @abstractmethod
    def cumulants(self, T: float) -> Cumulants:
        """
        Evaluate the cumulants of the model at a given time. This is useful e.g. to figure out integration bounds etc
        during pricing
        :param T: float, time to maturity (time at which cumulants are evaluated)
        :return: Cumulants object
        """
        raise NotImplementedError

    @abstractmethod
    def chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        raise NotImplementedError

