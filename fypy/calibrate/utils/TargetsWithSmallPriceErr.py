from typing import Callable, Optional

import numpy as np

from fypy.fit.Objective import Objective


class TargetsWithSmallPriceErr(Objective):
    def __init__(self,
                 targets: np.ndarray,
                 function: Callable[..., np.ndarray],
                 weights: Optional[np.ndarray] = None,
                 strength: float = 1.0,
                 small_price_penalty_mult: int = 3):
        """
        An objective representing a set of price targets which penalizes more heavily when a small price is missed
        from below (very small prices lead to instability in implied vol calculation, so we prefer to prevent that)
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
        self._min_target = np.min(self._targets) * 3
        self._function = function
        self._small_price_penalty_mult = small_price_penalty_mult

    def value(self) -> np.ndarray:
        """ Evaluate the targets objective, returns residuals per target """
        model_prices = self._function()
        resids = model_prices - self._targets
        resids[model_prices < self._min_target] *= self._small_price_penalty_mult

        return self._weights * resids

    def relative_error(self, min_price: 0.01, include_weights: bool = False) -> np.ndarray:
        """ Evaluate the targets objective, returns residuals per target """
        model_prices = self._function()

        err = self._weights * (model_prices - self._targets) / np.maximum(min_price, self._targets) \
            if include_weights else (model_prices - self._targets) / np.maximum(min_price, self._targets)

        return err