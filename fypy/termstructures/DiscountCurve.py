from abc import ABC, abstractmethod
from typing import Union
import numpy as np


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
