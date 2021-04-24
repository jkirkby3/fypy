from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from typing import List, Tuple, Optional, Union


class NIG(LevyModel):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 alpha: float = 15.,
                 beta: float = -5.,
                 delta: float = 0.5):
        """
        Normal Inverse Gaussian (NIG) model
        :param forwardCurve: ForwardCurve term structure
        :param alpha: float, tail/steepness parameter controlling the kurtosis (larger alpha=lighter tails)
        :param beta: float, skewness param: β <0 (resp.β >0) means the distribution is skewed
                to the left (resp.  theright), and β= 0 means the distribution is symmetric
        :param delta: float, scale parameter and plays an analogous role to the σ parameter of BSM or VG
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)
        self._params = np.asarray([alpha, beta, delta])

    # ================
    # Model Parameters
    # ================

    @property
    def alpha(self) -> float:
        """  Volatility param (similar to black scholes)"""
        return self._params[0]

    @property
    def beta(self) -> float:
        """ Symmetry param (theta=0 is symmetric distribution) """
        return self._params[1]

    @property
    def delta(self) -> float:
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
        asq = self.alpha ** 2
        bsq = self.beta ** 2
        temp = np.sqrt(asq - bsq)
        w = self.delta * (np.sqrt(asq - (self.beta + 1) ** 2) - temp)  # convexity correction
        rn_drift = self.forwardCurve.drift(0, 1) + w

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * (rn_drift + self.delta * self.beta / temp),
                         c2=T * (self.delta * asq * (asq - bsq) ** (-1.5)),
                         c4=T * 3 * self.delta * asq * (asq + 4 * bsq) * (asq - bsq) ** (-3.5)
                         )

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        asq = self.alpha ** 2
        bsq = self.beta ** 2
        temp = np.sqrt(asq - bsq)
        w = self.delta * (np.sqrt(asq - (self.beta + 1) ** 2) - temp)  # convexity correction
        rn_drift = self.forwardCurve.drift(0, 1) + w

        return 1j * xi * rn_drift - self.delta * (np.sqrt(asq - (self.beta + 1j * xi) ** 2) - temp)

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
        # TODO: add constraint that beta \in (-alpha, alpha-1)
        return [(0, np.inf), (-np.inf, np.inf), (0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([15., -5., 0.5])
