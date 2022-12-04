from fypy.model.FourierModel import Cumulants, FourierModel
from fypy.model.sv.Heston import _HestonBase
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from abc import ABC, abstractmethod

import numpy as np
from typing import List, Tuple, Optional, Union


class _HestonJumpsBase(_HestonBase, ABC):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve):
        """
        Base Heston + Jumps model class. This model is an instance of a Fourier model, SLV, etc.
        This model extends Heston by adding jumps (of some type)
        :param forwardCurve: EquityForward curve object, contains all term structure info
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)

    # =============================
    # Fourier Interface Implementation
    # =============================

    @abstractmethod
    def cumulants(self, T: float) -> Cumulants:
        """
        Evaluate the cumulants of the model at a given time. This is useful e.g. to figure out integration bounds etc
        during pricing
        :param T: float, time to maturity (time at which cumulants are evaluated)
        :return: Cumulants object
        """
        raise NotImplementedError

    @abstractmethod
    def _jump_chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function of jump component
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        raise NotImplementedError

    def chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        return self._heston_chf(T, xi) * self._jump_chf(T, xi)

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 5 + self._num_jump_params()

    @abstractmethod
    def _num_jump_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _jump_param_bounds(self) -> List[Tuple]:
        raise NotImplementedError

    @abstractmethod
    def _jump_default_params(self) -> List[float]:
        raise NotImplementedError

    def param_bounds(self) -> Optional[List[Tuple]]:
        # v_0, theta, kappa, sigma_v, rho
        return self._heston_param_bounds() + self._jump_param_bounds()

    def default_params(self) -> Optional[np.ndarray]:
        """
        v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6
        :return: np.ndarray, the default parameters for this model (e.g. a reasonable starting guess for calibration)
        """
        return np.asarray(self._heston_default_params() + self._jump_default_params())


class Bates(_HestonJumpsBase):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6,
                 lam: float = 0.2,
                 muj: float = 0.0,
                 sigj: float = 0.2
                 ):
        """
        Bates model class. This model is an instance of a Fourier model, SLV, etc.
        This model extends Heston by adding lognormal jumps (ie a hybrid of Heston and Merton models)
        :param forwardCurve: EquityForward curve object, contains all term structure info
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)
        self._params = np.asarray([v_0, theta, kappa, sigma_v, rho, lam, muj, sigj])

    # ================
    # Model Parameters (See _HestonBase for first 5 parameters)
    # ================

    @property
    def lam(self) -> float:
        """ """
        return self._params[5]

    @property
    def muj(self) -> float:
        """  """
        return self._params[6]

    @property
    def sigj(self) -> float:
        """  """
        return self._params[7]

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
        cumulants = self._heston_cumulants(T)

        sigj2 = .5 * self.sigj ** 2
        w_jump = - self.lam * (np.exp(self.muj + sigj2) - 1)  # convexity correction for jump component
        cumulants.rn_drift += w_jump
        cumulants.c1 += T * self.lam * self.muj
        cumulants.c2 += T * self.lam * (self.muj ** 2 + self.sigj ** 2)
        cumulants.c4 += T * (self.lam * (self.muj ** 4 + 6 * self.sigj ** 2 * self.muj ** 2 + 3 * self.sigj ** 4))

        return cumulants

    def _jump_chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function of jump component
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        sigj2 = .5 * self.sigj ** 2
        return np.exp(T * self.lam * (np.exp(1j * xi * self.muj - sigj2 * xi ** 2) - 1))

    # =============================
    # Calibration Interface Implementation
    # =============================

    def _num_jump_params(self) -> int:
        return 3

    def _jump_param_bounds(self) -> List[Tuple]:
        return [(0, np.inf), (-np.inf, np.inf), (0, np.inf)]

    def _jump_default_params(self) -> List[float]:
        """"""
        # lam, muj, sigj
        return [0.05, 0.0, 0.2]
