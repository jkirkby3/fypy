from abc import ABC, abstractmethod

import numpy as np


class StrikesPricer(ABC):
    """
    Abstract class for pricing a homogeneous type of instrument (e.g. European options),
    which are distinguished according to Time, Strike, and call or put.
    For example, we can price multiple strikes efficiently under a "Fourier" model such as Levy or Heston using
    a Fast Fourier Transform pricer. These typically have efficiencies for prices a set of strikes with common maturity

    NOTE: you must define at least one of price_strikes() or price(), and the other will be defined.
    For efficiency, you may wish to override both, depending on the type of pricer you implement

    Note: implementations of the strikes pricer is will target a single type of option, e.g. European, American, etc.
    """
    def price_strikes(self,
                      T: float,
                      K: np.ndarray,
                      is_calls: np.ndarray) -> np.ndarray:
        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param is_calls: np.array[bool], indicators of if strikes are calls (true) or puts (false)
        :return: np.array, prices of strikes
        """
        output = np.empty_like(K, dtype=float)
        self.price_strikes_fill(T=T, K=K, is_calls=is_calls, output=output)
        return output

    def price_strikes_fill(self,
                           T: float,
                           K: np.ndarray,
                           is_calls: np.ndarray,
                           output: np.ndarray):
        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param is_calls: np.ndarray[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :return: None, this method fills in the output array, make sure its sized properly first
        """
        if len(output) != len(is_calls) or len(K) != len(is_calls):
            raise ValueError("You supply")
        for i in range(len(K)):
            output[i] = self.price(T, K[i], is_calls[i])

    def price(self, T: float, K: float, is_call: bool) -> float:
        """
        Price a single strike (of whatever type of instrument the strikes pricer can price)

        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        prices = self.price_strikes(T=T, K=np.asarray([K]), is_calls=np.asarray([is_call]))
        return prices[0]
