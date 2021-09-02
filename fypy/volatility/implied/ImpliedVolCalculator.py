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

        # Compute Implied Volsv
        for i in range(len(vols)):
            vols[i] = self.imply_vol(price=prices[i],
                                     strike=strikes[i],
                                     is_call=is_calls[i],
                                     ttm=ttm)

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
        cp = 1 if is_call else -1
        disc = self._disc_curve(ttm)
        fwd = self._fwd_curve(ttm)
        return implied_volatility_from_a_transformed_rational_guess(price/disc, fwd, strike, ttm, cp)


if __name__ == '__main__':
    from fypy.termstructures.EquityForward import EquityForward
    from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
    spot = 100
    disc_curve = DiscountCurve_ConstRate(rate=0.0)
    fwd_curve = EquityForward(S0=spot, discount=disc_curve)
    ivc = ImpliedVolCalculator_Black76(fwd_curve=fwd_curve, disc_curve=disc_curve)

    p = 0.05
    K_ = 100
    T = 0.5
    v = ivc.imply_vol(price=p, strike=K_, ttm=T, is_call=True)
    p2 = black76_price(fwd_curve(T), K_, True, v, disc_curve(T), T)

    diff = p2 - p
    print(diff)