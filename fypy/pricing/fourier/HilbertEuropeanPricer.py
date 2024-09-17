from fypy.termstructures.EquityForward import EquityForward
from fypy.model.FourierModel import FourierModel
import numpy as np
from scipy.fft import fft
from fypy.pricing.StrikesPricer import StrikesPricer


class HilbertEuropeanPricer(StrikesPricer):
    def __init__(
        self,
        model: FourierModel,
        alpha: float = 0.75,
        eta: float = 0.1,
        N: int = 2**9,
        Nh: int = 2**5,
    ):
        """Hilbert method for Pricing European options under a Fourier model (i.e. using ChF)


        Args:
            model (FourierModel): FourierModel, model to price under
            alpha (float, optional):  Defaults to 0.75.
            eta (float, optional):  Defaults to 0.1.
            N (int, optional):  Defaults to 2**9.
            Nh (int, optional):  Defaults to 2**9.
        """
        self._model = model
        self._alpha = alpha
        self._eta = eta
        self._N = N
        self._Nh = Nh
        self._h = 2 * np.pi / Nh
        self._logS0 = np.log(self._model.spot())

    def price(self, T: float, K: float, is_call: bool) -> float:
        """
        Price a single strike of European option
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        gridL = np.arange(-int(self._N / 2), 0)
        gridR = -gridL[::-1]
        H = (
            np.sum(
                self._g(self._h * gridL, T, K) * (np.cos(np.pi * gridL) - 1) / gridL
                + self._g(self._h * gridR, T, K) * (np.cos(np.pi * gridR) - 1) / gridR
            )
            / np.pi
        )
        disc = self._model.discountCurve(T)
        price = 0.5 * np.real(
            self._model.forwardCurve(T) * disc - K * disc + 1j * disc * H
        )
        if not is_call:
            price = price - (self._model.forwardCurve(T) * disc - K * disc)
        return price

    def _g(self, xi: np.ndarray, T: float, K: float):
        return np.exp(-1j * xi * np.log(K / self._model.spot())) * (
            self._model.spot() * self._model.chf(T, xi - 1j)
            - K * self._model.chf(T, xi)
        )
