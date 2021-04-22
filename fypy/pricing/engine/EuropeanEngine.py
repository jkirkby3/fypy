from fypy.pricing.engine.Engine import Engine
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.instrument.Instrument import VanillaOption
import numpy as np


class EuropeanEngine(Engine):
    def __init__(self, strikesPricer: StrikesPricer):
        """
        European pricing Engine, which can price European options.
        :param strikesPricer: an instance of a StrikesPricer, e.g. a Fourier based pricer
        """
        self._strikesPricer = strikesPricer

    def price_instrument(self, inst: VanillaOption) -> float:
        """
        Price a vanilla instrument (implements the Engine interface)
        :param inst: Instrument object, the instrument to price
        :return: price of instrument
        """
        return self._strikesPricer.price(T=inst.ttm, K=inst.strike, is_call=inst.is_call)

    def price_strikes(self,
                      T: float,
                      K: np.ndarray,
                      is_calls: np.ndarray) -> np.ndarray:
        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes
        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param is_calls: np.array[bool], indicators of if strikes are calls (true) or puts (false)
        :return: np.array, prices of strikes
        """
        return self._strikesPricer.price_strikes(T=T, K=K, is_calls=is_calls)

    def price(self, T: float, K: float, is_call: bool):
        """
        Price a single strike of European option
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        return self._strikesPricer.price(T=T, K=K, is_call=is_call)
