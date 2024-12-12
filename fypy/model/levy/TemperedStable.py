"""Implementation of Tempered Stable model"""

from typing import List, Tuple, Optional, Union
import numpy as np
from scipy.special import gamma

from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve


class TemperedStable(LevyModel):
    """
    Tempered Stable Model as introduced in:
    Tempered stable distributions and processes, U. Kuchler and S. Tappe
    """

    def __init__(
        self,
        forwardCurve: ForwardCurve,
        discountCurve: DiscountCurve,
        alpha_p: float = 0.2,
        beta_p: float = 0.5,
        lambda_p: float = 1,
        alpha_m: float = 0.3,
        beta_m: float = 0.3,
        lambda_m: float = 2,
    ):
        super().__init__(
            forwardCurve=forwardCurve,
            discountCurve=discountCurve,
            params=np.asarray([alpha_p, beta_p, lambda_p, alpha_m, beta_m, lambda_m]),
        )

    ####################################
    ####### MODEL  PARAMETERS ##########
    ####################################

    @property
    def alpha_p(self) -> float:
        """Model Parameter"""
        return self._params[0]

    @property
    def beta_p(self) -> float:
        """Model Parameter"""
        return self._params[1]

    @property
    def lambda_p(self) -> float:
        """Model Parameter"""
        return self._params[2]

    @property
    def alpha_m(self) -> float:
        """Model Parameter"""
        return self._params[3]

    @property
    def beta_m(self) -> float:
        """Model Parameter"""
        return self._params[4]

    @property
    def lambda_m(self) -> float:
        """Model Parameter"""
        return self._params[5]

    ####################################
    ####### CUMULANTS HELPERS ##########
    ####################################

    def cumulants(self, T: float) -> Cumulants:
        """
        Evaluate the cumulants of the model at a given time.
        This is useful e.g. to figure out integration bounds etc
        during pricing
        :param T: float, time to maturity (time at which cumulants are evaluated)
        :return: Cumulants object
        """
        alpha_p, beta_p, lambda_p = self.alpha_p, self.beta_p, self.lambda_p
        alpha_m, beta_m, lambda_m = self.alpha_m, self.beta_m, self.lambda_m

        rn_drift = self.risk_neutral_log_drift()

        def cumulants_gen(n: int):
            return gamma(n - beta_p) * alpha_p / (lambda_p ** (n - beta_p)) + (
                -1
            ) ** n * gamma(n - beta_m) * alpha_m / (lambda_m ** (n - beta_m))

        return Cumulants(
            T=T,
            rn_drift=rn_drift,
            c1=T * (rn_drift + cumulants_gen(1)),
            c2=T * cumulants_gen(2),
            c4=T * cumulants_gen(3),
        )

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via:
        chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at
        input points in frequency domain
        """
        alpha_p, beta_p, lambda_p = self.alpha_p, self.beta_p, self.lambda_p
        alpha_m, beta_m, lambda_m = self.alpha_m, self.beta_m, self.lambda_m
        rn_drift = self.risk_neutral_log_drift()

        return (
            1j * xi * rn_drift
            + alpha_p
            * gamma(-beta_p)
            * ((lambda_p - 1j * xi) ** beta_p - lambda_p**beta_p)
            + alpha_m
            * gamma(-beta_m)
            * ((lambda_m + 1j * xi) ** beta_m - lambda_m**beta_m)
        )

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model,
        added to log process drift to ensure
        risk neutrality
        """
        alpha_p, beta_p, lambda_p = self.alpha_p, self.beta_p, self.lambda_p
        alpha_m, beta_m, lambda_m = self.alpha_m, self.beta_m, self.lambda_m
        return -(
            alpha_p * gamma(-beta_p) * ((lambda_p - 1) ** beta_p - lambda_p**beta_p)
            + alpha_m * gamma(-beta_m) * ((lambda_m + 1) ** beta_m - lambda_m**beta_m)
        )

    #############################################
    ### Calibration Interface Implementation  ###
    #############################################

    def num_params(self) -> int:
        return 6

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf), (0, 1), (0, np.inf), (0, np.inf), (0, 1), (0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([1, 0.5, 1, 1, 0.3, 1])
