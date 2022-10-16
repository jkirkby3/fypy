from typing import Callable, Optional

import numpy as np

from fypy.fit.Objective import Objective


class Targets(Objective):
    def __init__(self,
                 targets: np.ndarray,
                 function: Callable[..., np.ndarray],
                 weights: Optional[np.ndarray] = None,
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
        if weights is None:
            weights = np.ones_like(targets) / len(targets)
        self._weights = self._strength * np.sqrt(weights)
        self._targets = targets
        self._function = function

    def value(self) -> np.ndarray:
        """ Evaluate the targets objective, returns residuals per target """
        return self._weights * (self._function() - self._targets)

    def relative_error(self, min_price: 0.01, include_weights: bool = False) -> np.ndarray:
        """ Evaluate the targets objective, returns residuals per target """
        return self._weights * (self._function() - self._targets) / np.maximum(min_price, self._targets)\
            if include_weights else (self._function() - self._targets) / np.maximum(min_price, self._targets)
