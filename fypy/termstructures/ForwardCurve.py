from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class ForwardCurve(ABC):
    """
    Abstract base class for deterministic forward curves.
    Examples:
        Equity:    F(T) = S_0 * Div(T) / Disc(T)   (more generally includes dividends, borrow cost, etc.)
        FX:        F(T) = FX_0 * Disc_f(T) / Disc_d(T)
        Rates:     F(T) = IBOR(T), the forward rate for some IBOR curve, e.g. LIBOR 3M
        Commodity: F(T) = Futures(T), ie. some interpolation of the futures curve
    """

    @abstractmethod
    def spot(self) -> float:
        """ Spot price. In some cases this is the actual spot (e.g. Equity/FX), otherwise it is F(0) """
        raise NotImplementedError

    @abstractmethod
    def fwd_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Forward at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, forward(s) at time(s) in the future
        """
        raise NotImplementedError

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Forward at time T in the future.  Ability to call term structure using ()
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, forward(s) at time(s) in the future
        """
        return self.fwd_T(T)

    def drift(self, t: float, T: float) -> float:
        """
        Drift implied by the forward curve, implied over a time interval [t,T]
        :param t: float, start time
        :param T: float, end time
        :return: float, drift implied over [t,T]
        """
        T = max(T, t + 1e-09)
        return np.log(self.fwd_T(T) / self.fwd_T(t)) / (T - t)
