from abc import ABC, abstractmethod
import numpy as np


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


