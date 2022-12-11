from fypy.model.FourierModel import Cumulants
from fypy.model.sv.Bates import _HestonJumpsBase
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve

import numpy as np
from typing import List, Tuple, Optional, Union


class HestonDEJumps(_HestonJumpsBase):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6,
                 lam: float = 3.,
                 p_up: float = 0.2,
                 eta1: float = 25.,
                 eta2: float = 10.
                 ):
        """
        Heston + Double Exponential Jumps model class. This model is an instance of a Fourier model, SLV, etc.
        This model extends Heston by adding double exponential jumps (ie a hybrid of Heston and Kou DE Jump Diffusion)
        :param forwardCurve: EquityForward curve object, contains all term structure info
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve)
        self._params = np.asarray([v_0, theta, kappa, sigma_v, rho, lam, p_up, eta1, eta2])

    # ================
    # Model Parameters (See _HestonBase for first 5 parameters)
    # ================

    @property
    def lam(self) -> float:
        """ """
        return self._params[5]

    @property
    def p_up(self) -> float:
        """ Probablity that an arriving jump is up (vs down) """
        return self._params[6]

    @property
    def eta1(self) -> float:
        """ Left Tail heaviness parameter (for down jumps) """
        return self._params[7]

    @property
    def eta2(self) -> float:
        """ Right Tail heaviness parameter (for up jumps) """
        return self._params[8]

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

        lam, p_up, eta1, eta2 = self.lam, self.p_up, self.eta1, self.eta2

        w_jump = - lam * (p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)  # convexity correction

        cumulants.rn_drift += w_jump
        cumulants.c1 += T * (lam * p_up / eta1 + lam * (1 - p_up) / eta2)
        cumulants.c2 += T * (2 * lam * p_up / (eta1 ** 2) + 2 * lam * (1 - p_up) / (eta2 ** 2))
        cumulants.c4 += T * (24 * lam * (p_up / eta1 ** 4 + (1 - p_up) / eta2 ** 4))

        return cumulants

    def _jump_chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function of jump component
        :param T: float, time to maturity
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, characteristic function evaluated at input points in frequency domain
        """
        lam, p_up, eta1, eta2 = self.lam, self.p_up, self.eta1, self.eta2

        w = - lam * (p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)  # convexity correction

        temp2 = lam * ((1 - p_up) * eta2 / (eta2 + 1j * xi) + p_up * eta1 / (eta1 - 1j * xi) - 1)
        return np.exp(T * (temp2 + 1j * xi * w))

    # =============================
    # Calibration Interface Implementation
    # =============================

    def _num_jump_params(self) -> int:
        return 4

    def _jump_param_bounds(self) -> List[Tuple]:
        return [(0, np.inf), (0, 1), (0, np.inf), (0, np.inf)]

    def _jump_default_params(self) -> List[float]:
        """"""
        # lam, p_up, eta1, eta2
        return [0.05, 0.2, 25., 10.]
