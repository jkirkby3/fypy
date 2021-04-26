from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from typing import List, Tuple, Optional, Union


class MertonJD(LevyModel):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 sigma: float = 0.2,
                 lam: float = 0.2,
                 muj: float = 0.0,
                 sigj: float = 0.2):
        """
        Merton's Jump Diffusion (MJD) model. Jumps arrive as Poisson process, and magnitude of jumps is normal
        :param forwardCurve: ForwardCurve term structure
        :param sigma: float, volatility param (similar to black scholes)
        :param lam: float, arrival rate of Poisson Jumps
        :param muj: float, mean of jump size
        :param sigj: float, standard deviation param of jump size
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve,
                         params=np.asarray([sigma, lam, muj, sigj]))

    # ================
    # Model Parameters
    # ================

    @property
    def sigma(self) -> float:
        """  Volatility param (similar to black scholes)"""
        return self._params[0]

    @property
    def lam(self) -> float:
        """ """
        return self._params[1]

    @property
    def muj(self) -> float:
        """  """
        return self._params[2]

    @property
    def sigj(self) -> float:
        """  """
        return self._params[3]

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
        sig2 = .5 * self.sigma ** 2
        sigj2 = .5 * self.sigj ** 2
        w = -sig2 - self.lam * (np.exp(self.muj + sigj2) - 1)  # convexity correction
        rn_drift = self.forwardCurve.drift(0, 1) + w

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * (rn_drift + self.lam * self.muj),
                         c2=T * self.lam * (self.sigma ** 2 / self.lam + self.muj ** 2 + self.sigj ** 2),
                         c4=T * self.lam * (self.muj ** 4 + 6 * self.sigj ** 2 * self.muj ** 2 + 3 * self.sigj ** 4)
                         )

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        sig2 = .5 * self.sigma ** 2
        sigj2 = .5 * self.sigj ** 2
        w = -sig2 - self.lam * (np.exp(self.muj + sigj2) - 1)  # convexity correction
        rn_drift = self.forwardCurve.drift(0, 1) + w

        return 1j * xi * rn_drift - sig2 * xi ** 2 + self.lam * (np.exp(1j * xi * self.muj - sigj2 * xi ** 2) - 1)

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 4

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf), (0, np.inf), (-np.inf, np.inf), (0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([0.2, 0.2, 0.0, 0.2])
