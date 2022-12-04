from fypy.model.FourierModel import Cumulants, FourierModel
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from abc import ABC, abstractmethod

import numpy as np
from typing import List, Tuple, Optional, Union


class _HestonBase(FourierModel, ABC):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve):
        """
        Base Heston model class, used for Heston and Heston Jump models (Bates)
        :param forwardCurve: EquityForward curve object, contains all term structure info
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)
        self._params: Optional[np.ndarray] = None

    # =============================
    # Model Parameters
    # =============================

    @property
    def v_0(self) -> float:
        """ Initial Variance param """
        return self._params[0]

    @property
    def theta(self) -> float:
        """ Long Term Variance param """
        return self._params[1]

    @property
    def kappa(self) -> float:
        """ Mean-Reversion rate param """
        return self._params[2]

    @property
    def sigma_v(self) -> float:
        """ Vol of Variance param """
        return self._params[3]

    @property
    def rho(self) -> float:
        """ Correlation (between vol and asset innovations) param """
        return self._params[4]

    # =============================
    # Fourier Interface Implementation
    # =============================

    def _heston_cumulants(self, T: float) -> Cumulants:
        """
        Evaluate the cumulants of the model at a given time. This is useful e.g. to figure out integration bounds etc
        during pricing
        :param T: float, time to maturity (time at which cumulants are evaluated)
        :return: Cumulants object
        """
        v_0, theta, kappa, sigma_v, rho = self.v_0, self.theta, self.kappa, self.sigma_v, self.rho

        w = -0.5 * theta  # convexity correction
        rn_drift = self.forwardCurve.drift(0, T) + w
        c1 = T * rn_drift + (1 - np.exp(-kappa * T)) * (theta - v_0) / (2 * kappa)

        c2 = 1 / (8 * kappa ** 3) * (
                sigma_v * T * kappa * np.exp(-kappa * T) * (v_0 - theta) * (8 * kappa * rho - 4 * sigma_v)
                + kappa * rho * sigma_v * (1 - np.exp(-kappa * T)) * (16 * theta - 8 * v_0)
                + 2 * theta * kappa * T * (-4 * kappa * rho * sigma_v + sigma_v ** 2 + 4 * kappa ** 2)
                + sigma_v ** 2 * ((theta - 2 * v_0) * np.exp(-2 * kappa * T) + theta
                                  * (6 * np.exp(-kappa * T) - 7) + 2 * v_0)
                + 8 * kappa ** 2 * (v_0 - theta) * (1 - np.exp(-kappa * T)))

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=c1,
                         c2=c2,
                         c4=0)

    def _heston_chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        v_0, theta, kappa, sigma_v, rho = self.v_0, self.theta, self.kappa, self.sigma_v, self.rho

        alpha = -.5 * (xi * xi + xi * 1j)
        beta = kappa - rho * sigma_v * xi * 1j
        omega2 = sigma_v ** 2
        gamma = .5 * omega2

        D = np.sqrt(beta ** 2 - 4.0 * alpha * gamma)

        bD = beta - D
        eDt = np.exp(-D * T)

        G = bD / (beta + D)
        B = (bD / omega2) * ((1.0 - eDt) / (1.0 - G * eDt))
        psi = (1.0 - G * eDt) / (1.0 - G)
        A = ((kappa * theta) / omega2) * (bD * T - 2.0 * np.log(psi))

        drift = self.forwardCurve.drift(0, T)
        return np.exp(A + B * v_0 + 1j * xi * drift * T)

    # =============================
    # Calibration Interface Implementation
    # =============================

    def set_params(self, params: np.ndarray):
        self._params = params
        return self

    def get_params(self) -> np.ndarray:
        return self._params

    def _heston_default_params(self) -> List[float]:
        # v_0, theta, kappa, sigma_v, rho
        return [0.04, 0.04, 2., 0.3, -0.6]

    def _heston_param_bounds(self) -> Optional[List[Tuple]]:
        # v_0, theta, kappa, sigma_v, rho
        return [(0.0001, np.inf), (0.0001, np.inf), (0., 50.), (0.00001, 10.), (-1., 1.)]


class Heston(_HestonBase):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6):
        """
        Heston model class. This model is an instance of a Fourier model, SLV, etc.
        :param forwardCurve: EquityForward curve object, contains all term structure info
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)
        self._params = np.asarray([v_0, theta, kappa, sigma_v, rho])

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
        return self._heston_cumulants(T)

    def chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        return self._heston_chf(T, xi)

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 5

    def param_bounds(self) -> Optional[List[Tuple]]:
        # v_0, theta, kappa, sigma_v, rho
        return self._heston_param_bounds()

    def default_params(self) -> Optional[np.ndarray]:
        """
        v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6
        :return:
        """
        # v_0, theta, kappa, sigma_v, rho
        return np.asarray(self._heston_param_bounds())
