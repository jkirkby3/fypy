from fypy.model.levy.LevyModel import LevyModel
from fypy.model.FourierModel import Cumulants
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
import numpy as np
from typing import List, Tuple, Optional, Union


class _BilateralGammaBase(LevyModel):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 params: np.ndarray):
        """
        Bilateral Gamma (BG) model
        :param forwardCurve: ForwardCurve term structure
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve, params=params)

    # ================
    # Model Parameters
    # ================

    @property
    def alpha_p(self) -> float:
        """"""
        return self._params[0]

    @property
    def lambda_p(self) -> float:
        """"""
        return self._params[1]

    @property
    def alpha_m(self) -> float:
        """"""
        return self._params[2]

    @property
    def lambda_m(self) -> float:
        """"""
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
        alpha_p, lam_p, alpha_m, lam_m = self.alpha_p, self.lambda_p, self.alpha_m, self.lambda_m

        rn_drift = self.risk_neutral_log_drift()

        def factorial(n):
            if n <= 1:
                return 1
            if n == 2:
                return 2
            if n == 3:
                return 6
            raise NotImplementedError

        # Using equation (2.8)

        def cumulant(n: int):
            return factorial(n - 1) * (alpha_p / lam_p ** n + (-1) ** n * alpha_m / lam_m ** n)

        return Cumulants(T=T,
                         rn_drift=rn_drift,
                         c1=T * rn_drift,  # + cumulant(1)
                         c2=T * cumulant(2),
                         c4=T * cumulant(4))

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        alpha_p, lam_p, alpha_m, lam_m = self.alpha_p, self.lambda_p, self.alpha_m, self.lambda_m
        return -np.log((lam_p / (lam_p - 1)) ** alpha_p * (lam_m / (lam_m + 1)) ** alpha_m)

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        alpha_p, lam_p, alpha_m, lam_m = self.alpha_p, self.lambda_p, self.alpha_m, self.lambda_m

        rn_drift = self.risk_neutral_log_drift()

        return 1j * xi * rn_drift \
               + np.log((lam_p / (lam_p - 1j * xi)) ** alpha_p * (lam_m / (lam_m + 1j * xi)) ** alpha_m)

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 4

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf), (1, np.inf), (0, np.inf), (0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([1.18, 10.57, 1.44, 5.57])


class BilateralGamma(_BilateralGammaBase):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 alpha_p: float,
                 lambda_p: float,
                 alhpa_m: float,
                 lambda_m: float):
        """
        Bilateral Gamma (BG) model
        :param forwardCurve: ForwardCurve term structure
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve,
                         params=np.asarray([alpha_p, lambda_p, alhpa_m, lambda_m]))


class BilateralGammaMotion(_BilateralGammaBase):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 alpha_p: float,
                 lambda_p: float,
                 alhpa_m: float,
                 lambda_m: float,
                 sigma: float):
        """
        Bilateral Gamma Motion (BGM) model, an extension of Bilateral Gamma model to include Brownian motion
        component, resulting in significantly better calibration and elimimation of smile kinks produced by
        Bilateral Gamma

        Ref: "The Bilateral Gamma Motion: Calibration and Option Pricing", JL Kirkby, CA Rinella, J-P Aguilar, (2023)

        :param forwardCurve: ForwardCurve term structure
        """
        super().__init__(forwardCurve=forwardCurve, discountCurve=discountCurve,
                         params=np.asarray([alpha_p, lambda_p, alhpa_m, lambda_m, sigma]))

    # ================
    # Model Parameters
    # ================
    @property
    def sigma(self) -> float:
        return self._params[4]

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
        cumulants = super().cumulants(T)
        cumulants.c2 += self.sigma ** 2

        return cumulants

    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        return super().convexity_correction() - 0.5 * self.sigma ** 2

    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T*symbol(xi)),  for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        return super().symbol(xi=xi) - 0.5 * self.sigma ** 2 * xi * xi

    # =============================
    # Calibration Interface Implementation
    # =============================

    def num_params(self) -> int:
        return 5

    def param_bounds(self) -> Optional[List[Tuple]]:
        return [(0, np.inf), (1, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]

    def default_params(self) -> Optional[np.ndarray]:
        return np.asarray([1.18, 10.57, 1.44, 5.57, 0.01])
