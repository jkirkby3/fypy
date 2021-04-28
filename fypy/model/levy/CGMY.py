from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from scipy.special import gamma
from typing import List, Tuple, Optional, Union


class CMGY(LevyModel):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 C: float = 0.02,
                 G: float = 5.,
                 M: float = 15.,
                 Y: float = 1.2):
        """
        Carr-Geman-Madan-Yor (CGMY) model.  When Y=0, this model reduces to VG
        :param forwardCurve: ForwardCurve term structure
        :param C: float, viewed as a measure of the overall level of activity, and influences kurtosis
        :param G: float, rate of exponential decay on the right tail
        :param M: float, rate of exponential decay on the left tail. Typically for equities G < M, ie the left
            tail is then heavier than the right (more down risk)
        :param Y: float, controls the "fine structure" of the process
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve,
                         params=np.asarray([C, G, M, Y]))

    # ================
    # Model Parameters
    # ================

    @property
    def C(self) -> float:
        """ Model Parameter """
        return self._params[0]

    @property
    def G(self) -> float:
        """ Model Parameter """
        return self._params[1]

    @property
    def M(self) -> float:
        """ Model Parameter  """
        return self._params[2]

    @property
    def Y(self) -> float:
        """ Model Parameter  """
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
        C, G, M, Y = self.C, self.G, self.M, self.Y
        rn_drift = self.risk_neutral_log_drift()

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * (rn_drift + C * gamma(1 - Y) * (M ** (Y - 1) - G ** (Y - 1))),
                         c2=T * C * gamma(2 - Y) * (M ** (Y - 2) + G ** (Y - 2)),
                         c4=T * C * gamma(4 - Y) * (M ** (Y - 4) + G ** (Y - 4))
                         )

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        C, G, M, Y = self.C, self.G, self.M, self.Y
        rn_drift = self.risk_neutral_log_drift()

        return 1j * xi * rn_drift + C * gamma(-Y) * ((M - 1j * xi) ** Y - M ** Y + (G + 1j * xi) ** Y - G ** Y)

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        C, G, M, Y = self.C, self.G, self.M, self.Y
        return -C * gamma(-Y) * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)  # convexity correction

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 4

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf), (0, np.inf), (0, np.inf), (-np.inf, 2)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([0.02, 5, 15, 1.2])
