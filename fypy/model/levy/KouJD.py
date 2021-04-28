from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from typing import List, Tuple, Optional, Union


class KouJD(LevyModel):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 sigma: float = 0.15,
                 lam: float = 3.,
                 p_up: float = 0.2,
                 eta1: float = 25.,
                 eta2: float = 10.):
        """
        Kou's double exponential jump diffusion model
        :param forwardCurve: ForwardCurve term structure
        :param sigma: float,
        :param lam: float,
        :param p_up: float,
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve,
                         params=np.asarray([sigma, lam, p_up, eta1, eta2]))

    # ================
    # Model Parameters
    # ================

    @property
    def sigma(self) -> float:
        """  Volatility param (similar to black scholes)"""
        return self._params[0]

    @property
    def lam(self) -> float:
        """ Poisson Jump arrival rate """
        return self._params[1]

    @property
    def p_up(self) -> float:
        """ Probablity that an arriving jump is up (vs down) """
        return self._params[2]

    @property
    def eta1(self) -> float:
        """ Left Tail heaviness parameter (for down jumps) """
        return self._params[3]

    @property
    def eta2(self) -> float:
        """ Right Tail heaviness parameter (for up jumps) """
        return self._params[4]

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
        sigma, lam, p_up, eta1, eta2 = self.sigma, self.lam, self.p_up, self.eta1, self.eta2
        rn_drift = self.risk_neutral_log_drift()

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * (rn_drift + lam * p_up / eta1 + lam * (1 - p_up) / eta2),
                         c2=T * (sigma ** 2 + 2 * lam * p_up / (eta1 ** 2) + 2 * lam * (1 - p_up) / (eta2 ** 2)),
                         c4=T * (24 * lam * (p_up / eta1 ** 4 + (1 - p_up) / eta2 ** 4))
                         )

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        sigma, lam, p_up, eta1, eta2 = self.sigma, self.lam, self.p_up, self.eta1, self.eta2
        rn_drift = self.risk_neutral_log_drift()

        sig2 = .5 * sigma ** 2
        temp2 = -sig2 * xi ** 2 + lam * ((1 - p_up) * eta2 / (eta2 + 1j * xi) + p_up * eta1 / (eta1 - 1j * xi) - 1)

        return 1j * xi * rn_drift + temp2

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        sigma, lam, p_up, eta1, eta2 = self.sigma, self.lam, self.p_up, self.eta1, self.eta2
        w = -.5 * sigma ** 2 - lam * (
                p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)  # convexity correction
        return w

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 5

    def param_bounds(self) -> Optional[List[Tuple]]:
        # sigma, lam, p_up, eta1, eta2
        return [(0, np.inf), (0, np.inf), (0, 1), (0, np.inf), (0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([0.15, 3., 0.2, 25., 10.])
