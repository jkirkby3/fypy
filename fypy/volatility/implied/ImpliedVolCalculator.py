from abc import ABC, abstractmethod
from typing import List
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from fypy.pricing.analytical.black_scholes import *


class ImpliedVolCalculator(ABC):
    """ Base class for implied volatiliy calculators on standard options """
    @abstractmethod
    def imply_vol(self,
                  price: float,
                  strike: float,
                  is_call: bool,
                  ttm: float,
                  disc: float,
                  underlying: float) -> float:
        """
        Calculate implied volatility from a single price
        :param price: float, option price
        :param strike: float, strike price
        :param is_call: bool, true for call, else put
        :param ttm: float, Time to maturity (in years)
        :param disc: float, Discount factor (e.g. 0.98)
        :param underlying: float, the spot or forward (depends on the calculator type)
        :return: implied vol (float)
        """
        raise NotImplementedError

    def imply_vols(self,
                   strikes: np.ndarray,
                   prices: np.ndarray,
                   is_call: Union[np.ndarray, List[int]],
                   ttm: float,
                   disc: float,
                   underlying: float) -> np.ndarray:
        """
        Calculates bid and ask implied vols from bid and ask prices, for a single time to maturity
        Note: you can override this with a vectorized version if available
        :param strikes: np.ndarray, strikes per option
        :param prices: np.ndarray, prices per option (same order as strikes)
        :param is_call: np.ndarray of bools, true for call, false for put
        :param ttm: float, Time to maturity (common to all options)
        :param disc: float, Discount factor for this maturity
        :param underlying: float, the underlying (spot or forward, consistent with the calculator)
        :return: np.ndarray of implied vols
        """
        vols = np.zeros_like(prices)

        # Compute Implied Volsv
        for i in range(len(vols)):
            vols[i] = self.imply_vol(price=prices[i],
                                     strike=strikes[i],
                                     is_call=is_call[i],
                                     disc=disc, underlying=underlying, ttm=ttm)

        return vols


class ImpliedVolCalculator_Black76(ImpliedVolCalculator):
    """
    Implied vol calculator that operates on Black76 option prices, ie underlying is forward. To apply with Black-Scholes
    on the spot, underlying = spot*Div(T)*Disc(T)
    """
    def imply_vol(self,
                  price: float,
                  strike: float,
                  is_call: bool,
                  ttm: float,
                  disc: float,
                  underlying: float) -> float:
        """ Note: the underlying is forward for Black76 """
        cp = 1 if is_call else -1
        return implied_volatility_from_a_transformed_rational_guess(price/disc, underlying, strike, ttm, cp)


if __name__ == '__main__':

    ivc = ImpliedVolCalculator_Black76()
    p = 0.05
    K_ = 100
    F = 100
    T = 0.5
    D = np.exp(-0.5*T)
    v = ivc.imply_vol(price=p, strike=K_, underlying=F, disc=D, ttm=T, is_call=True)
    p2 = black76_price(F, K_, True, v, D, T)

    diff = p2 - p
    print(diff)