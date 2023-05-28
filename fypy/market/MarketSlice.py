import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod

from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator


class Strike(ABC):
    @abstractmethod
    def strike(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def is_call(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def mid_price(self) -> Optional[float]:
        raise NotImplementedError


class StrikeFilter(ABC):
    @abstractmethod
    def keep_strike(self, strike: Strike) -> bool:
        raise NotImplementedError


class OTMStrikeFilter(StrikeFilter):
    def keep_strike(self, strike: Strike) -> bool:
        if strike.is_call() and strike.forward() < strike.strike():
            return True
        if not strike.is_call() and strike.forward() >= strike.strike():
            return True

        return False


class RelativeStrikeFilter(StrikeFilter):
    def __init__(self, min_relative: float, max_relative: float):
        self._min_relative = min_relative
        self._max_relative = max_relative

    def keep_strike(self, strike: Strike) -> bool:
        rel_strike = strike.strike() / strike.forward()
        if rel_strike > self._max_relative:
            return False
        if rel_strike < self._min_relative:
            return False

        return True


class MidPriceFilter(StrikeFilter):
    def __init__(self, min_price: float):
        self._min_price = min_price

    def keep_strike(self, strike: Strike) -> bool:
        if strike.mid_price() < self._min_price:
            return False

        return True


class StrikeFilters(StrikeFilter):
    def __init__(self, filters: Optional[List[StrikeFilter]] = None):
        self._filters = filters or []

    def add_filter(self, slice_filter: StrikeFilter):
        self._filters.append(slice_filter)

    def keep_strike(self, strike: Strike) -> bool:
        for strike_filter in self._filters:
            if not strike_filter.keep_strike(strike):
                return False
        return True


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

        # Implied volatilities (these can be set/filled after initialization)
        self.bid_vols: Optional[np.ndarray] = None
        self.mid_vols: Optional[np.ndarray] = None
        self.ask_vols: Optional[np.ndarray] = None

    def set_vols(self,
                 bid_vols: Optional[np.ndarray] = None,
                 mid_vols: Optional[np.ndarray] = None,
                 ask_vols: Optional[np.ndarray] = None):
        """
        Set the implied volatilities from their value. Alternatively, you can fill them by supplying an
        ImpliedVolCalculator
        :param bid_vols: np.ndarray, bid implied vols (optional)
        :param mid_vols: np.ndarray, mid implied vols (optional)
        :param ask_vols: np.ndarray, ask implied vols (optional)
        :return: None
        """
        self.bid_vols = bid_vols
        self.ask_vols = ask_vols
        self.mid_vols = mid_vols

    def fill_implied_vols(self, calculator: ImpliedVolCalculator):
        """
        Fill the implied vols given a calculator. Fills in for each of bid,mid,ask, but only those that have
        corresponding prices
        :param calculator: ImpliedVolCalculator, a calculator used to fill in the vols from prices
        :return: None
        """
        for prices, which in zip((self.bid_prices, self.mid_prices, self.ask_prices),
                                 ('bid', 'mid', 'ask')):
            if prices is not None:
                vols = calculator.imply_vols(strikes=self.strikes, prices=prices, is_calls=self.is_calls, ttm=self.T)
                if which == 'bid':
                    self.bid_vols = vols
                elif which == 'mid':
                    self.mid_vols = vols
                else:
                    self.ask_vols = vols

    class SliceStrike(Strike):
        def __init__(self,
                     index: int,
                     market_slice: 'MarketSlice'):
            self._index = index
            self._market_slice = market_slice

        def strike(self) -> float:
            return self._market_slice.strikes[self._index]

        def forward(self) -> float:
            return self._market_slice.F

        def is_call(self) -> int:
            return self._market_slice.is_calls[self._index]

        def mid_price(self) -> Optional[float]:
            return self._market_slice.mid_prices[self._index] if self._market_slice.mid_prices is not None else None

    def filter_strikes(self, strike_filter: StrikeFilter):
        strikes = []
        bid_prices = []
        mid_prices = []
        ask_prices = []
        is_calls = []

        bid_vols = []
        mid_vols = []
        ask_vols = []

        for i in range(len(self.strikes)):
            strike = MarketSlice.SliceStrike(index=i, market_slice=self)
            if strike_filter.keep_strike(strike):
                strikes.append(self.strikes[i])

                if self.bid_prices is not None:
                    bid_prices.append(self.bid_prices[i])

                if self.mid_prices is not None:
                    mid_prices.append(self.mid_prices[i])

                if self.ask_prices is not None:
                    ask_prices.append(self.ask_prices[i])

                if self.bid_vols is not None:
                    bid_vols.append(self.bid_vols[i])

                if self.mid_vols is not None:
                    mid_vols.append(self.mid_vols[i])

                if self.ask_vols is not None:
                    ask_vols.append(self.ask_vols[i])

                is_calls.append(self.is_calls[i])

        new_slice = MarketSlice(T=self.T,
                                F=self.F,
                                disc=self.disc,
                                strikes=np.asarray(strikes),
                                bid_prices=np.asarray(bid_prices),
                                mid_prices=np.asarray(mid_prices),
                                ask_prices=np.asarray(ask_prices),
                                is_calls=np.asarray(is_calls, dtype=int))
        new_slice.set_vols(bid_vols=np.asarray(bid_vols) if bid_vols else None,
                           mid_vols=np.asarray(mid_vols) if mid_vols else None,
                           ask_vols=np.asarray(ask_vols) if ask_vols else None)

        return new_slice

    def _set_prices(self):
        if self.mid_prices is None:
            if self.bid_prices is None or self.ask_prices is None:
                raise ValueError("If you dont supply mid prices, must supply bid and ask prices")
            self.mid_prices = (self.bid_prices + self.ask_prices) / 2


if __name__ == '__main__':
    # Example usage
    from fypy.pricing.analytical.black_scholes import black76_price_strikes
    from fypy.termstructures.EquityForward import EquityForward, DiscountCurve_ConstRate
    from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76

    strikes_ = np.arange(50, 150, 1)
    is_calls_ = np.ones(len(strikes_), dtype=bool)
    T = 1.

    disc_curve = DiscountCurve_ConstRate(rate=0.02)
    fwd = EquityForward(S0=100, discount=disc_curve)

    prices_ = black76_price_strikes(F=fwd(T), K=strikes_, is_calls=is_calls_, vol=0.2, disc=disc_curve(T), T=T)

    mkt_slice = MarketSlice(T=T, F=fwd(T), disc=disc_curve(T), strikes=strikes_, is_calls=is_calls_,
                            mid_prices=prices_)

    ivc = ImpliedVolCalculator_Black76(fwd_curve=fwd, disc_curve=disc_curve)
    mkt_slice.fill_implied_vols(calculator=ivc)
    vols = mkt_slice.mid_vols

    print(len(mkt_slice.strikes))

    filtered = mkt_slice.filter_strikes(strike_filter=OTMStrikeFilter())
    print(len(filtered.strikes))
