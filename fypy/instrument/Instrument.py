from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class Instrument(ABC):
    """
    Base instrument class.
    """

    def __init__(self):
        pass


# TODO: add excercise class, vanilla option will take an excerise


class VanillaOption(Instrument):
    def __init__(self,
                 strike: float,
                 ttm: float,
                 is_call: bool):
        """ Vanilla Option """
        super().__init__()
        self._strike = strike
        self._ttm = ttm
        self._is_call = is_call

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def ttm(self) -> float:
        return self._ttm

    @property
    def is_call(self) -> bool:
        return self._is_call

    def payoff(self, underlying: Union[float, np.array]) -> Union[float, np.array]:
        """
        Calculate the payoff (cashflow) that would occur given a particular value of the underlying
        :param underlying: float or array of underlying value
        :return: float or array, matches type that was supplied
        """
        return np.maximum(0, underlying - self._strike) if self._is_call \
            else np.maximum(0, self._strike - underlying)
