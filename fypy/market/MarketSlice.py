import numpy as np
from typing import Optional


class MarketSlice(object):
    def __init__(self,
                 T: float,
                 F: float,
                 disc: float,
                 strikes: np.ndarray,
                 is_calls: np.ndarray,
                 bid_prices: Optional[np.ndarray] = None,
                 mid_prices: Optional[np.ndarray] = None,
                 ask_prices: Optional[np.ndarray] = None):
        """
        Container holding market option prices for a single slice (tenor) in a price surface. You must either supply
        mid prices, or bid and ask prices
        :param T: float, time to maturity for this slice
        :param F: float, the foward price, F(T)
        :param disc: float, the discount, D(T)
        :param strikes: np.ndarray, the strikes in increasing order along the slice- no sorting is done, this is assumed
        :param is_calls: np.ndarray, true/false indicator per strike, true if its a call option
        :param bid_prices: np.ndarray, optional, bid prices per strike
        :param mid_prices: np.ndarray, optional, mid prices per strike (if not supplied, supply both bid and ask)
        :param ask_prices: np.ndarray, optional, ask prices per strike
        """
        self.T = T
        self.F = F  # forward price  (used to determine main quotes)
        self.disc = disc
        self.strikes = strikes
        self.is_calls = is_calls

        self.bid_prices = bid_prices
        self.mid_prices = mid_prices
        self.ask_prices = ask_prices
        self._set_prices()

    def _set_prices(self):
        if self.mid_prices is None:
            if self.bid_prices is None or self.ask_prices is None:
                raise ValueError("If you dont supply mid prices, must supply bid and ask prices")
            self.mid_prices = (self.bid_prices + self.ask_prices) / 2
