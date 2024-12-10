"""Implementation of the BGIG model for PROJ framework"""

from typing import List, Tuple, Optional, Union

import numpy as np
import scipy

from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve


class BGIG(LevyModel):
    """
    Implementation of the BGIG model as introduced in:

    The bilateral generalized inverse Gaussian process with applications
    to financial modeling, G. AGAZZOTTI, JP. Aguilar
    """

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
        BGIG model

        Args:
            forwardCurve (ForwardCurve): fwd
            discountCurve (DiscountCurve): discount
            a_p (float, optional): Defaults to 500.
            b_p (float, optional): Defaults to 0.05.
            p_p (float, optional): Defaults to 2.
            a_m (float, optional): Defaults to 300.
            b_m (float, optional): Defaults to 0.03.
            p_m (float, optional): Defaults to 2.
        """
        super().__init__(
            forwardCurve=forwardCurve,
            discountCurve=discountCurve,
            params=np.asarray([a_p, b_p, p_p, a_m, b_m, p_m]),
        )

    ####################################
    ####### MODEL  PARAMETERS ##########
    ####################################

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

    ####################################
    ####### CUMULANTS HELPERS ##########
    ####################################

    def ratio_bessel(self, omega: float, p: float) -> float:
        """
        ratio of bessel function

        Args:
            omega (float): omega
            p (float): p params

        Returns:
            float: ratio of bessel
        """
        return scipy.special.kv(p + 1, omega) / scipy.special.kv(p, omega)

    def c1(self, omega: float, eta: float, p: float) -> float:
        """
        cumulants of order 1 of a one sided BIG distribution

        Args:
            omega (float): omega
            eta (float): eta
            p (float): p

        Returns:
           float: c1
        """
        return self.ratio_bessel(omega, p) * eta

    def c2(self, omega: float, eta: float, p: float) -> float:
        """
        cumulants of order 2 of a one sided BIG distribution

        Args:
            omega (float): omega
            eta (float): eta
            p (float): p

        Returns:
           float: c2
        """
        polynom = (
            -(self.ratio_bessel(omega, p) ** 2)
            + (2 * (p + 1) / omega) * self.ratio_bessel(omega, p)
            + 1
        )
        return polynom * eta**2

    def c3(self, omega: float, eta: float, p: float):
        """
        cumulants of order 3 of a one sided BIG distribution

        Args:
            omega (float): omega
            eta (float): eta
            p (float): p

        Returns:
           float: c3
        """
        polynom = (
            2 * self.ratio_bessel(omega, p) ** 3
            - (6 * (p + 1) / omega) * self.ratio_bessel(omega, p) ** 2
            + ((4 * (p + 1) * (p + 2) / omega**2) - 2) * self.ratio_bessel(omega, p)
            + 2 * (p + 1) / omega
        )
        return polynom * eta**3

    def c4(self, omega: float, eta: float, p: float):
        """
        cumulants of order 4 of a one sided BIG distribution

        Args:
            omega (float): omega
            eta (float): eta
            p (float): p

        Returns:
           float: c4
        """
        polynom = (
            2 * self.ratio_bessel(omega, p) ** 3
            - (6 * (p + 1) / omega) * self.ratio_bessel(omega, p) ** 2
            + ((4 * (p + 1) * (p + 2) / omega**2) - 2) * self.ratio_bessel(omega, p)
            + 2 * (p + 1) / omega
        )
        return polynom * eta**4

    def cumulants_gen(
        self,
        order: int,
    ) -> float:
        """
        compute cumulants of order "order"

        Args:
            order (int): order of the cumulant

        Raises:
            NotImplementedError: if order > 4

        Returns:
            cumulant (float)
        """
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
        Evaluate the cumulants of the model at a given time.
        This is useful e.g. to figure out integration bounds etc
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
        Levy symbol, uniquely defines Characteristic Function via:
        chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
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
        Computes the convexity correction for the Levy model,
        added to log process drift to ensure
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

    #############################################
    ### Calibration Interface Implementation  ###
    #############################################

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
