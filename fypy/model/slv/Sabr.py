from fypy.fit.Calibratable import Calibratable
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve

import numpy as np
from typing import List, Tuple, Optional, Union


class Sabr(Calibratable):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 v_0: float = 0.04,
                 alpha: float = 0.5,
                 beta: float = 0.9,
                 rho: float = -0.6
                 ):
        """
        SABR stochastic local volatility model class
        :param discountCurve: Discount curve term structure
        :param forwardCurve: Forward curve term structure
        """

        self._params = np.asarray([v_0, alpha, beta, rho])

        self._discountCurve = discountCurve
        self._forwardCurve = forwardCurve

    @property
    def v_0(self) -> float:
        return self._params[0]

    @property
    def alpha(self) -> float:
        return self._params[1]

    @property
    def beta(self) -> float:
        return self._params[2]

    @property
    def rho(self) -> float:
        return self._params[3]

    def implied_vol(self,
                    K: float,
                    T: float,
                    fwd: float
                    ):
        F0 = fwd
        if F0 == K:
            K += 1e-07

        nu, alpha, beta, rho = self.v_0, self.alpha, self.beta, self.rho

        z = (alpha / (nu * (1 - beta))) * (F0 ** (1 - beta) - K ** (1 - beta))
        chiz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))

        FK = F0 * K

        sig0 = alpha * np.log(F0 / K) / chiz
        sig1 = ((1 - beta) * nu) ** 2 / 24 / FK ** (1 - beta) + 0.25 * (rho * beta * alpha * nu) / FK ** (
                (1 - beta) / 2) + (2 - 3 * rho ** 2) / 24 * alpha ** 2

        return sig0 * (1.0 + sig1 * T)

    def spot(self) -> float:
        """
        Get the spot. In the case where a spot doesn't really make sense, it can simply return the default
        implementation, which is forward at time = 0
        """
        return self._forwardCurve.spot()

    @property
    def discountCurve(self) -> DiscountCurve:
        """ Get the discount Curve term structure """
        return self._discountCurve

    @property
    def forwardCurve(self) -> ForwardCurve:
        """ Get the forwarding term structure """
        return self._forwardCurve

    def set_params(self, params: np.ndarray):
        """
        Sets the parameters in calibratable
        :param params: np.array, the new parameters
        :return: self
        """
        self._params = params

    def get_params(self) -> np.ndarray:
        """
        Get the current set of params.
        Note: set params
        :return: np.array, current parameters
        """
        return self._params

    def num_params(self) -> int:
        """
        Number of parameters that are settable / calibratable
        :return: int, num params
        """
        return 4

    def param_bounds(self) -> Optional[List[Tuple]]:
        """
        Theoretical parameter bounds. These dont have to be what you used during optimization, but providing reasonable
        bounds here allows a model to be fit out-of-the-box
        :return: list of tuples, upper and lower bounds on each parameter.
        """
        # v_0, alpha, beta, rho
        return [(0.00001, np.inf), (0.00001, 100), (0., 1.0), (-0.999999, 0.999999)]

    def default_params(self) -> Optional[np.ndarray]:
        """
        Default set of parameters that can be used as an initial guess, in the absence of better information.
        These dont have to be what you used during optimization (e.g. by inspecting market data first),
        but providing reasonable parameters here allows a model to be fit out-of-the-box
        :return: array of default parameters, by default returns None, to indicate no default parameters provided
        """
        # v_0, alpha, beta, rho
        return np.asarray([0.04, 0.5, 0.9, -0.6])
