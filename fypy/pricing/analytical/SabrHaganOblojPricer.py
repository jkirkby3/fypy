from fypy.model.slv.Sabr import Sabr
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.pricing.analytical.black_scholes import black76_price


class SabrHaganOblojPricer(StrikesPricer):
    """
    Sabr strikes pricer using the implied vol approximation of Hagan-Obloj
    """
    def __init__(self, model: Sabr):
        self._model = model

    def price(self, T: float, K: float, is_call: bool) -> float:
        """
        Price a single strike (of whatever type of instrument the strikes pricer can price)

        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        fwd = self._model.forwardCurve.fwd_T(T)
        disc = self._model.discountCurve.discount_T(T)
        implied_vol = self._model.implied_vol(K=K, T=T, fwd=fwd)
        return black76_price(T=T, F=fwd, K=K, is_call=is_call, vol=implied_vol, disc=disc)
