from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from typing import List, Tuple, Optional, Union


class VarianceGamma(LevyModel):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 sigma: float = 0.2,
                 theta: float = 0.0,
                 nu: float = 0.8):
        """
        Variance Gamma (VG) model
        :param forwardCurve: ForwardCurve term structure
        :param sigma: float, volatility param (similar to black scholes)
        :param theta: float, symmetry param (theta=0 is symmetric distribution)
        :param nu: float, tail heaviness param (small nu -> heavier tail)
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)
        self._params = np.asarray([sigma, theta, nu])

    # ================
    # Model Parameters
    # ================

    @property
    def sigma(self) -> float:
        """  Volatility param (similar to black scholes)"""
        return self._params[0]

    @property
    def theta(self) -> float:
        """ Symmetry param (theta=0 is symmetric distribution) """
        return self._params[1]

    @property
    def nu(self) -> float:
        """ Tail heaviness param (small nu -> heavier tail) """
        return self._params[2]

    # =============================
    # Fourier Interface Implementation
    # =============================

    def cumulants(self, T: float) -> Cumulants:
        """
        Evaluate the cumulants of the model at a given time. This is useful e.g. to figure out integration bounds etc
        during pricing
        :param T: float, time to maturity (time at which cumulants are evaluated)
        :return: Cumulants object
        """
        sig2 = self.sigma ** 2
        thet2 = self.theta * self.theta
        nu = self.nu
        w = np.log(1 - self.theta * nu - 0.5 * sig2 * nu) / nu  # convexity correction
        rn_drift = self.forwardCurve.drift(0, T) + w

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * (rn_drift + self.theta),
                         c2=T * (sig2 + nu * thet2),
                         c4=T * 3 * (sig2 * sig2 * nu + 2 * thet2 * thet2 * nu ** 3 + 4 * sig2 * thet2 * nu * nu))

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        sig2 = .5 * self.sigma ** 2
        nu = self.nu
        w = np.log(1 - self.theta * nu - sig2 * nu) / nu  # convexity correction
        rn_drift = self.forwardCurve.drift(0, 1) + w

        return 1j * xi * rn_drift - np.log(1 - 1j * self.theta * nu * xi + sig2 * nu * xi ** 2) / nu

    # =============================
    # Calibration Interface Implementation
    # =============================

    def set_params(self, params: np.ndarray):
        self._params = params
        return self

    def get_params(self) -> np.ndarray:
        return self._params

    def num_params(self) -> int:
        return 3

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([0.2, -.01, 0.8])
