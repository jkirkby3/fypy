from typing import Union
import numpy as np
from fypy.termstructures.DiscountCurve import DiscountCurve, DiscountCurve_ConstRate
from fypy.termstructures.ForwardCurve import ForwardCurve


class EquityForward(ForwardCurve):
    def __init__(self,
                 S0: float,
                 discount: DiscountCurve,
                 divDiscount: DiscountCurve = DiscountCurve_ConstRate(rate=0)):
        """
        Equity forward curve: spot driven, deterministic forward
        :param S0: float, spot at reference time
        :param discount: DiscountCurve, e.g. OIS discounting curve to calculate NPV of cashflows
        :param divDiscount: DiscountCurve, dividend discounting curve, e.g. exp(-q*T)
        """
        self._S0 = S0
        self._discount = discount
        self._divDiscount = divDiscount

    def spot(self) -> float:
        """ Get the spot """
        return self._S0

    @property
    def discountCurve(self) -> DiscountCurve:
        """ Access the discount curve, e.g. OIS discounting curve to calculate NPV of cashflows"""
        return self._discount

    @property
    def divDiscountCurve(self) -> DiscountCurve:
        """ Access the dividend discounting curve, e.g. exp(-q*T) """
        return self._divDiscount

    @staticmethod
    def from_rates(S0: float, r: float, q: float):
        """
        Create an equity forward from constant interest rate and dividend yield (common modeling case)
        :param S0: float, spot
        :param r: float, interest rate
        :param q: float, div yield
        :return: EquityForward
        """
        return EquityForward(S0=S0,
                             discount=DiscountCurve_ConstRate(rate=r),
                             divDiscount=DiscountCurve_ConstRate(rate=q))

    def fwd_T(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Forward at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, forward(s) at time(s) in the future
        """
        return self._S0 * self._divDiscount(T) / self._discount(T)

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Forward at time T in the future
        :param T: float or np.ndarray, time(s) in the future
        :return: float or np.ndarray, forward(s) at time(s) in the future
        """
        return self.fwd_T(T)
