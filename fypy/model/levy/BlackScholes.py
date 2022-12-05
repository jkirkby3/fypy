import numpy as np
from typing import List, Tuple, Optional, Union

from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.process.LocalVolProcess1D import LocalVolProcess1D
from fypy.process.Drift import Drift_FC


class BlackScholes(LevyModel, LocalVolProcess1D):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 sigma: float = 0.2):
        """
        Black-Scholes model class. This model is an instance of a LevyModel, Diffusion, etc.
        :param forwardCurve: EquityForward curve object, contains all term structure info
        :param sigma: float, volatility
        """
        LevyModel.__init__(self, forwardCurve=forwardCurve, discountCurve=discountCurve,
                           params=np.asarray([sigma, ]))
        LocalVolProcess1D.__init__(self, drift=Drift_FC(fwd=forwardCurve))

    # =============================
    # Model Parameters
    # =============================

    @property
    def vol(self) -> float:
        """ Volatility param """
        return self._params[0]

    # =============================
    # LocalVolProcess Interface
    # =============================
    def sigma_LV_dt(self, S: Union[float, np.ndarray], t: float, dt: float) -> Union[float, np.ndarray]:
        """
        Local volatility component:  sigma(S,t):= sigma_LV(S,t)*S(t)
        :param S: float or array, underlying level
        :param t: float, time of evaluation
        :param dt: float, the time step size, we average the vol over [t, t+dt)
        :return: float or array, matches input S
        """
        if isinstance(S, float):
            return self.vol

        return np.full_like(S, self.vol)

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
        sig2 = self.vol ** 2
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
        sig2 = 0.5 * self.vol ** 2
        return xi * (1j * self.risk_neutral_log_drift() - sig2 * xi)

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        return -0.5 * self.vol ** 2

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 1

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([0.2, ])
