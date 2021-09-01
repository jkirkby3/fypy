from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class Objective(ABC):
    def __init__(self, strength: float = 1.0):
        """
        A generic calibration objective, e.g. the price targets or some penalty to combat overfitting, arbitrage, etc.
        A calibration problem is a set of objectives, each given a strength. The calibration will weigh these objectives
        relative to their strengths, to determine the optimal parameters
        :param strength: float, the strength of this particular objective/penalty
        """
        self._strength = strength

    @property
    def strength(self) -> float:
        return self._strength

    @abstractmethod
    def value(self) -> np.ndarray:
        """ Evaluate the objective function, return the array of residuals """
        raise NotImplementedError


class Targets(Objective):
    def __init__(self,
                 weights: np.ndarray,
                 targets: np.ndarray,
                 function: Callable[..., np.ndarray],
                 strength: float = 1.0):
        """
        An objective representing a set of targets (e.g. market prices, volatilities, etc)
        :param weights: np.ndarray, the weight to apply per target
        :param targets: np.ndarray, the targets themselves (e.g. market prices)
        :param function: function, evaluated at each of the targets, determines the residual
        :param strength: float, the strength of this particular objective
        """
        super().__init__(strength=strength)
        # take sqrt since they are applied to residual before squaring.
        # The strength is in space of sqrt(sum(squares)), so let it be squared with the resituals
        self._weights = self._strength * np.sqrt(weights)
        self._targets = targets
        self._function = function

    def value(self) -> np.ndarray:
        """ Evaluate the targets objective, returns residuals per target """
        return self._weights * (self._function() - self._targets)
