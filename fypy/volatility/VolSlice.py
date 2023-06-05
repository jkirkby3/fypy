from abc import ABC, abstractmethod
from typing import Union, Callable
from scipy.interpolate import interp1d

import numpy as np

from fypy.market.MarketSlice import MarketSlice
from fypy.volatility.StrikeConverter import StrikeConverter, LogRelativeStrikeConverter


class VolSlice(ABC):
    def __init__(self,
                 ttm: float,
                 fwd: float):
        self._ttm = ttm
        self._fwd = fwd

    @property
    def ttm(self) -> float:
        return self._ttm

    @property
    def fwd(self) -> float:
        return self._fwd

    @abstractmethod
    def vol(self, K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def total_var(self, K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.ttm * np.square(self.vol(K))


class ModelVolSlice(VolSlice):
    def __init__(self,
                 ttm: float,
                 fwd: float,
                 strike_converter: StrikeConverter = LogRelativeStrikeConverter()):
        super().__init__(ttm=ttm, fwd=fwd)
        self._strike_converter = strike_converter

    @property
    def strike_converter(self) -> StrikeConverter:
        return self._strike_converter

    def vol(self, K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.model_vol(self._strike_converter.convert(K=K, T=self.ttm, F=self.fwd))

    @abstractmethod
    def model_vol(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def model_var(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.ttm * np.square(self.model_vol(x))


class InterpolatedVolSlice(ModelVolSlice):
    def __init__(self,
                 ttm: float,
                 fwd: float,
                 interpolation: Callable,
                 strike_converter: StrikeConverter = LogRelativeStrikeConverter()
                 ):
        super().__init__(ttm=ttm, fwd=fwd)
        self._strike_converter = strike_converter

        # Note: we perform interpolation in model strike space (e.g. log relative strike space) using strike converter
        self._interp = interpolation

    @staticmethod
    def from_model_strikes(x: np.ndarray, vols: np.ndarray, ttm: float, fwd: float,
                           strike_converter: StrikeConverter = LogRelativeStrikeConverter()):
        # TODO: allow other interpolations
        interp = interp1d(x=x, y=vols, assume_sorted=True)
        return InterpolatedVolSlice(ttm=ttm,
                                    fwd=fwd,
                                    interpolation=interp,
                                    strike_converter=strike_converter)

    @staticmethod
    def from_market_strikes(K: np.ndarray, vols: np.ndarray, ttm: float, fwd: float,
                            strike_converter: StrikeConverter = LogRelativeStrikeConverter()):
        return InterpolatedVolSlice.from_model_strikes(ttm=ttm,
                                                       fwd=fwd,
                                                       x=strike_converter.convert(K=K, F=fwd, T=ttm),
                                                       strike_converter=strike_converter,
                                                       vols=vols)

    @staticmethod
    def from_market_slice(market_slice: MarketSlice,
                          strike_converter: StrikeConverter = LogRelativeStrikeConverter()
                          ) -> 'InterpolatedVolSlice':
        return InterpolatedVolSlice.from_market_strikes(ttm=market_slice.T,
                                                        fwd=market_slice.F,
                                                        vols=market_slice.mid_vols,
                                                        K=market_slice.strikes,
                                                        strike_converter=strike_converter)

    def model_vol(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._interp(x)
