from abc import ABC, abstractmethod
from typing import List
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.pricing.analytical.black_scholes import *


class ImpliedVolCalculator(ABC):
    """ Base class for implied volatiliy calculators on standard options """

    @abstractmethod
    def imply_vol(self,
                  price: float,
                  strike: float,
                  is_call: bool,
                  ttm: float) -> float:
        """
        Calculate implied volatility from a single price
        :param price: float, option price
        :param strike: float, strike price
        :param is_call: bool, true for call, else put
        :param ttm: float, Time to maturity (in years)
        :return: implied vol (float)
        """
        raise NotImplementedError

    def imply_vols(self,
                   strikes: np.ndarray,
                   prices: np.ndarray,
                   is_calls: Union[np.ndarray, List[int]],
                   ttm: float) -> np.ndarray:
        """
        Calculates bid and ask implied vols from bid and ask prices, for a single time to maturity
        Note: you can override this with a vectorized version if available
        :param strikes: np.ndarray, strikes per option
        :param prices: np.ndarray, prices per option (same order as strikes)
        :param is_calls: np.ndarray of bools, true for call, false for put
        :param ttm: float, Time to maturity (common to all options)
        :return: np.ndarray of implied vols
        """
        vols = np.zeros_like(prices)

        # Compute Implied Vols
        for i in range(len(vols)):
            try:
                vols[i] = self.imply_vol(price=prices[i],
                                         strike=strikes[i],
                                         is_call=is_calls[i],
                                         ttm=ttm)
            except Exception as e:
                vols[i] = np.nan

        return vols


class ImpliedVolCalculator_Black76(ImpliedVolCalculator):
    """
    Implied vol calculator that operates on Black76 option prices, ie underlying is forward. To apply with Black-Scholes
    on the spot, underlying = spot*Div(T)*Disc(T)
    """

    def __init__(self, fwd_curve: ForwardCurve, disc_curve: DiscountCurve):
        self._fwd_curve = fwd_curve
        self._disc_curve = disc_curve

    def imply_vol(self,
                  price: float,
                  strike: float,
                  is_call: bool,
                  ttm: float) -> float:
        """ Note: the underlying is forward for Black76 """
        return self._imply_vol(price=price, strike=strike, is_call=is_call, ttm=ttm,
                               disc=self._disc_curve(ttm),
                               fwd=self._fwd_curve(ttm))

    def imply_vols(self,
                   strikes: np.ndarray,
                   prices: np.ndarray,
                   is_calls: Union[np.ndarray, List[int]],
                   ttm: float) -> np.ndarray:
        """
        Calculates bid and ask implied vols from bid and ask prices, for a single time to maturity
        Note: you can override this with a vectorized version if available
        :param strikes: np.ndarray, strikes per option
        :param prices: np.ndarray, prices per option (same order as strikes)
        :param is_calls: np.ndarray of bools, true for call, false for put
        :param ttm: float, Time to maturity (common to all options)
        :return: np.ndarray of implied vols
        """
        vols = np.zeros_like(prices)

        # This version precomputes the forward and discount for efficiency
        disc = self._disc_curve(ttm)
        fwd = self._fwd_curve(ttm)

        # Compute Implied Vols
        for i in range(len(vols)):
            try:
                vols[i] = self._imply_vol(price=prices[i],
                                          strike=strikes[i],
                                          is_call=is_calls[i],
                                          ttm=ttm,
                                          disc=disc,
                                          fwd=fwd)
            except Exception as e:
                vols[i] = np.nan

        return vols

    def _imply_vol(self,
                   price: float,
                   strike: float,
                   is_call: bool,
                   ttm: float,
                   disc: float,
                   fwd: float
                   ):
        cp = 1 if is_call else -1
        return implied_volatility_from_a_transformed_rational_guess(price / disc, fwd, strike, ttm, cp)
