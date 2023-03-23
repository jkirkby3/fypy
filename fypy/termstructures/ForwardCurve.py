from abc import ABC, abstractmethod
from typing import Union, Callable, List
from fypy.termstructures.Interpolation import LogLinearInterpolation, interp1d
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

    def spot(self) -> float:
        """ Spot price. In some cases this is the actual spot (e.g. Equity/FX), otherwise it is F(0) """
        return self.fwd_T(0)

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


class FlatForwardCurve(ForwardCurve):
    def __init__(self,
                 F0: float):
        """
        Flat Forward curve class, which always returns a constant value
        """
        self._F0 = F0

    def spot(self) -> float:
        return self._F0

    def fwd_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._F0


class InterpolatedForwardCurve(ForwardCurve):
    def __init__(self,
                 interp: Callable):
        """
        Interpolated Forward curve class
        :param interp: an interpolation of x and y values, representing times and forwards
        """
        self._interp = interp

    def fwd_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._interp(T)

    @classmethod
    def from_linear(cls,
                    ttms: Union[np.ndarray, List],
                    forwards: Union[np.ndarray, List]) -> 'ForwardCurve':
        """
        Convenience method to construct a linearly interpolated forward curve
        :param ttms: array-like, the x-points, correspond to tenors (times to maturity)
        :param forwards: array-like, the y-points, correspond to forwards
        :return: ForwardCurve, interpolation of points and values
        """
        interp = interp1d(ttms, forwards, fill_value='extrapolate', bounds_error=False)
        return cls(interp=interp)

    @classmethod
    def from_log_linear(cls,
                        ttms: Union[np.ndarray, List],
                        forwards: Union[np.ndarray, List]) -> 'ForwardCurve':
        """
        Convenience method to construct a log-linearly interpolated forward curve.
        Note: this assumes that all forwards are positive
        :param ttms: array-like, the x-points, correspond to tenors (times to maturity)
        :param forwards: array-like, the y-points, correspond to forwards
        :return: FowardCurve, interpolation of points and values
        """
        interp = LogLinearInterpolation(points=ttms, values=forwards)
        return cls(interp=interp)