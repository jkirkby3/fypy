from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.EquityForward import EquityForward
import numpy as np
from typing import List, Tuple, Optional, Union


class BlackScholes(LevyModel):
    def __init__(self,
                 forwardCurve: EquityForward,
                 sigma: float = 0.2):
        """
        Black-Scholes model class. This model is an instance of a LevyModel, Diffusion, etc.
        :param forwardCurve: EquityForward curve object, contains all term structure info
        :param sigma: float, volatility
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=forwardCurve.discountCurve,
                         params=np.asarray([sigma]))

    # =============================
    # Model Parameters
    # =============================

    @property
    def sigma(self) -> float:
        """ Volatility param """
        return self._params[0]

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
        w = -0.5 * sig2  # convexity correction
        rn_drift = self.forwardCurve.drift(0, T) + w

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * rn_drift,
                         c2=T * sig2,
                         c4=0)

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        sig2 = 0.5 * self.sigma ** 2
        return xi * (1j * self.risk_neutral_log_drift() - sig2 * xi)

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        return -0.5 * self.sigma ** 2

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 1

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([0.2, ])
