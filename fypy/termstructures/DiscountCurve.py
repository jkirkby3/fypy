from abc import ABC, abstractmethod
from typing import Union, Callable, List
import numpy as np
from scipy.interpolate import interp1d

from fypy.termstructures.Interpolation import LogLinearInterpolation


class DiscountCurve(ABC):
    """
    Base class for discounting curves (for any discounting term structure: rates, dividends, etc.)

    Note: will add date based functionality, day counting, etc.
    For now, this class is time based, to support common modeling problems. In the future, especially when dealing
    with interest rate products, it will support date based arguments and day counting conventions.
    """

    def implied_rate(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Implied continuous rate at a particular time
        :param T: float or np.ndarray, time from which we imply the continuous rate, over [0,T]
        :return: float or np.ndarray (matches shape of input), the implied rate
        """
        return -np.log(self.discount_T(T)) / T

    @abstractmethod
    def discount_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Discount at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, discounts(s) at time(s) in the future
        """
        raise NotImplementedError

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Discount at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, discounts(s) at time(s) in the future
        """
        return self.discount_T(T)


class DiscountCurve_ConstRate(DiscountCurve):
    def __init__(self, rate: float):
        """
        Constant rate discount curve, exp(-r*T)
        :param rate: float, rate of discounting (e.g interest rate, div yield, etc)
        """
        super().__init__()
        self._r = rate

    def discount_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Discount at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, discounts(s) at time(s) in the future
        """
        return np.exp(-self._r * T)


class EmptyDiscountCurve(DiscountCurve):
    """ Empty discount curve, always returns 1.0. """

    def discount_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Discount at time T in the future, this version always returns 1.0
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, discounts(s) at time(s) in the future
        """
        return 1.0


class InterpolatedDiscountCurve(DiscountCurve):
    def __init__(self,
                 interp: Callable):
        """
        Interpolated Discount curve class
        :param interp: an interpolation of x and y values, representing times and discounts
        """
        self._interp = interp

    def discount_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._interp(T)

    @classmethod
    def from_linear(cls,
                    ttms: Union[np.ndarray, List],
                    discounts: Union[np.ndarray, List]) -> 'DiscountCurve':
        """
        Convenience method to construct a linearly interpolated discount curve
        :param ttms: array-like, the x-points, correspond to tenors
            Note: make sure that the zeroth point is zero
        :param discounts: array-like, the y-points, correspond to discounts
        :return: DiscountCurve, interpolation of points and values
        """
        if not np.abs(ttms[0]) <= 1e-14 or not np.abs(discounts[0] - 1.) <= 1e-14:
            raise ValueError("The first point in the interpolation must be (0, 1.0)")

        interp = interp1d(ttms, discounts, fill_value='extrapolate', bounds_error=False)
        return cls(interp=interp)

    @classmethod
    def from_log_linear(cls,
                        ttms: Union[np.ndarray, List],
                        discounts: Union[np.ndarray, List]) -> 'DiscountCurve':
        """
        Convenience method to construct a log-linearly interpolated discount curve
        :param ttms: array-like, the x-points, correspond to tenors
            Note: make sure that the zeroth point is zero
        :param discounts: array-like, the y-points, correspond to discounts
        :return: DiscountCurve, interpolation of points and values
        """
        if not np.abs(ttms[0]) <= 1e-14 or not np.abs(discounts[0] - 1.) <= 1e-14:
            raise ValueError("The first point in the interpolation must be (0, 1.0)")

        interp = LogLinearInterpolation(points=ttms, values=discounts)
        return cls(interp=interp)
