from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from typing import List, Tuple, Optional, Union

import scipy


class BGIG(LevyModel):
    def __init__(
        self,
        forwardCurve: ForwardCurve,
        discountCurve: DiscountCurve,
        a_p: float = 500,
        b_p: float = 0.05,
        p_p: float = 2,
        a_m: float = 300,
        b_m: float = 0.03,
        p_m: float = 2,
    ):
        """
        Carr-Geman-Madan-Yor (CGMY) model.  When Y=0, this model reduces to VG
        :param forwardCurve: ForwardCurve term structure
        :param C: float, viewed as a measure of the overall level of activity, and influences kurtosis
        :param G: float, rate of exponential decay on the right tail
        :param M: float, rate of exponential decay on the left tail. Typically for equities G < M, ie the left
            tail is then heavier than the right (more down risk)
        :param Y: float, controls the "fine structure" of the process
        """
        super().__init__(
            forwardCurve=forwardCurve,
            discountCurve=discountCurve,
            params=np.asarray([a_p, b_p, p_p, a_m, b_m, p_m]),
        )

    # ================
    # Model Parameters
    # ================

    @property
    def a_p(self) -> float:
        """Model Parameter"""
        return self._params[0]

    @property
    def b_p(self) -> float:
        """Model Parameter"""
        return self._params[1]

    @property
    def p_p(self) -> float:
        """Model Parameter"""
        return self._params[2]

    @property
    def a_m(self) -> float:
        """Model Parameter"""
        return self._params[3]

    @property
    def b_m(self) -> float:
        """Model Parameter"""
        return self._params[4]

    @property
    def p_m(self) -> float:
        """Model Parameter"""
        return self._params[5]

    # =============================
    # Fourier Interface Implementation
    # =============================

    ###################################
    ####### CUMULANTS HELPER ##########
    ###################################
    def R(self, omega: float, p: float) -> float:
        return scipy.special.kv(p + 1, omega) / scipy.special.kv(p, omega)

    def c1(self, omega: float, eta: float, p: float):
        return self.R(omega, p) * eta

    def c2(self, omega: float, eta: float, p: float):
        polynom = (
            -(self.R(omega, p) ** 2) + (2 * (p + 1) / omega) * self.R(omega, p) + 1
        )
        return polynom * eta**2

    def c3(self, omega: float, eta: float, p: float):
        polynom = (
            2 * self.R(omega, p) ** 3
            - (6 * (p + 1) / omega) * self.R(omega, p) ** 2
            + ((4 * (p + 1) * (p + 2) / omega**2) - 2) * self.R(omega, p)
            + 2 * (p + 1) / omega
        )
        return polynom * eta**3

    def c4(self, omega: float, eta: float, p: float):
        polynom = (
            2 * self.R(omega, p) ** 3
            - (6 * (p + 1) / omega) * self.R(omega, p) ** 2
            + ((4 * (p + 1) * (p + 2) / omega**2) - 2) * self.R(omega, p)
            + 2 * (p + 1) / omega
        )
        return polynom * eta**4

    def cumulants_gen(
        self,
        order: int,
    ):
        a_p, b_p, p_p = self.a_p, self.b_p, self.p_p
        a_m, b_m, p_m = self.a_m, self.b_m, self.p_m

        omega_p = (a_p * b_p) ** 0.5
        omega_m = (a_m * b_m) ** 0.5
        eta_p = (a_p / b_p) ** (-0.5)
        eta_m = (a_m / b_m) ** (-0.5)

        match order:
            case 1:
                return self.c1(omega_p, eta_p, p_p) - self.c1(omega_m, eta_m, p_m)
            case 2:
                return self.c2(omega_p, eta_p, p_p) + self.c2(omega_m, eta_m, p_m)
            case 3:
                return self.c3(omega_p, eta_p, p_p) - self.c3(omega_m, eta_m, p_m)
            case 4:
                return self.c4(omega_p, eta_p, p_p) + self.c4(omega_m, eta_m, p_m)
            case _:
                raise NotImplementedError

    def cumulants(self, T: float) -> Cumulants:
        """
        Evaluate the cumulants of the model at a given time. This is useful e.g. to figure out integration bounds etc
        during pricing
        :param T: float, time to maturity (time at which cumulants are evaluated)
        :return: Cumulants object
        """
        rn_drift = self.risk_neutral_log_drift()

        return Cumulants(
            T=T,
            rn_drift=rn_drift,
            c1=T * (rn_drift + self.cumulants_gen(1)),
            c2=T * self.cumulants_gen(2),
            c4=T * self.cumulants_gen(4),
        )

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        a_p, b_p, p_p = self.a_p, self.b_p, self.p_p
        a_m, b_m, p_m = self.a_m, self.b_m, self.p_m
        rn_drift = self.risk_neutral_log_drift()

        return 1j * xi * rn_drift + np.log(
            (a_p / (a_p - 2j * xi)) ** (p_p / 2)
            * scipy.special.kv(p_p, (b_p * (a_p - 2j * xi)))
            / scipy.special.kv(p_p, (b_p * a_p))
            * (a_m / (a_m + 2j * xi)) ** (p_m / 2)
            * scipy.special.kv(p_m, (b_m * (a_m - 2j * xi)))
            / scipy.special.kv(p_m, (b_m * a_m))
        )

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        a_p, b_p, p_p = self.a_p, self.b_p, self.p_p
        a_m, b_m, p_m = self.a_m, self.b_m, self.p_m

        return -np.log(
            (a_p / (a_p - 2)) ** (p_p / 2)
            * scipy.special.kv(p_p, (b_p * (a_p - 2)))
            / scipy.special.kv(p_p, (b_p * a_p))
            * (a_m / (a_m + 2)) ** (p_m / 2)
            * scipy.special.kv(p_m, (b_m * (a_m + 2)))
            / scipy.special.kv(p_m, (b_m * a_m))
        )

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 6

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [
            (2, np.inf),
            (0, np.inf),
            (-np.inf, np.inf),
            (0, np.inf),
            (0, np.inf),
            (-np.inf, np.inf),
        ]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([500, 0.05, 2, 300, 0.03, 2])
