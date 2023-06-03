from typing import Dict, List, Optional
from fypy.market.MarketSlice import MarketSlice, StrikeFilter
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.termstructures.EquityForward import EquityForward
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.volatility.implied import ImpliedVolCalculator

import numpy as np
from abc import ABC, abstractmethod


class SliceFilter(ABC):
    @abstractmethod
    def keep(self, market_slice: MarketSlice) -> bool:
        raise NotImplementedError


class NeverFilterSlice(SliceFilter):
    def keep(self, market_slice: MarketSlice) -> bool:
        return True


class FilterTTM(SliceFilter):
    def __init__(self,
                 min_ttm: float = 0.,
                 max_ttm: float = np.nan):
        self._min_ttm = min_ttm
        self._max_ttm = max_ttm

    def keep(self, market_slice: MarketSlice) -> bool:
        if market_slice.T < self._min_ttm:
            return False
        if market_slice.T > self._max_ttm:
            return False

        return True


class SliceFilters(SliceFilter):
    def __init__(self, filters: Optional[List[SliceFilter]] = None):
        self._filters = filters or []

    def add_filter(self, slice_filter: SliceFilter):
        self._filters.append(slice_filter)

    def keep(self, market_slice: MarketSlice) -> bool:
        for slice_filter in self._filters:
            if not slice_filter.keep(market_slice):
                return False
        return True


class MarketSurface(object):
    def __init__(self,
                 slices: Dict[float, MarketSlice] = None,
                 forward_curve: Optional[ForwardCurve] = None,
                 discount_curve: Optional[DiscountCurve] = None):
        """
        Container class for an option price surface, composed of individual market slices, one per tenor
        :param slices: dict: {float, MarketSlice}, contains all slices (you can add more later)
        :param forward_curve: Optional[ForwardCurve], a forward curve for the market.
        :param discount_curve: Optional[DiscountCurve], the discount curve.
        """
        self._slices = slices or {}
        self._forward_curve = forward_curve
        self._discount_curve = discount_curve

    def add_slice(self, ttm: float, market_slice: MarketSlice):
        """
        Add a new market slice (overwrites if same ttm already exists in surface)
        :param ttm: float, time to maturity of the slice (tenor)
        :param market_slice: MarketSlice, the market slice prices object
        :return: self
        """
        self._slices[ttm] = market_slice
        return self

    @property
    def slices(self) -> Dict[float, MarketSlice]:
        """ Access all slices """
        return self._slices

    @property
    def forward_curve(self) -> Optional[ForwardCurve]:
        """ Return the forward curve, if it exists. """
        return self._forward_curve

    @forward_curve.setter
    def forward_curve(self, fwd_curve: ForwardCurve):
        self._forward_curve = fwd_curve

    @property
    def eq_forward_curve(self) -> Optional[EquityForward]:
        """ If there is a forward curve, and it is an equity forward curve, return it. Otherwise, return None. """
        if isinstance(self._forward_curve, EquityForward):
            return self._forward_curve
        return None

    @property
    def spot(self) -> Optional[float]:
        """ If there is a forward curve, pull of the spot and return it, else return None. """
        return self._forward_curve(0.) if self._forward_curve else None

    @property
    def discount_curve(self) -> DiscountCurve:
        """ Return the discount curve, if it exists. """
        return self._discount_curve

    @discount_curve.setter
    def discount_curve(self, discount_curve: DiscountCurve):
        self._discount_curve = discount_curve

    @property
    def ttms(self):
        """ Get the ttms in the surface """
        return list(self._slices.keys())

    @property
    def num_slices(self) -> int:
        """ Get number of slice in surface """
        return len(self._slices)

    def fill_implied_vols(self, calculator: ImpliedVolCalculator):
        """
        Fill the implied vols given a calculator. Fills in for each of bid,mid,ask, but only those that have
        corresponding prices
        :param calculator: ImpliedVolCalculator, a calculator used to fill in the vols from prices
        :return: None
        """
        for slice_ in self.slices.values():
            slice_.fill_implied_vols(calculator)

    def filter_slices(self,
                      slice_filter: SliceFilter = NeverFilterSlice(),
                      strike_filter: Optional[StrikeFilter] = None) -> 'MarketSurface':
        filtered_surface = MarketSurface(forward_curve=self._forward_curve,
                                         discount_curve=self._discount_curve)
        for ttm, market_slice in self.slices.items():
            if slice_filter.keep(market_slice):
                if strike_filter:
                    filtered_surface.add_slice(ttm, market_slice=market_slice.filter_strikes(strike_filter))
                else:
                    filtered_surface.add_slice(ttm, market_slice=market_slice)

        return filtered_surface
