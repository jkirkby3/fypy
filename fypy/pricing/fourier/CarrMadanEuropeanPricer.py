from fypy.termstructures.EquityForward import EquityForward
from fypy.model.FourierModel import FourierModel
import numpy as np
from scipy.fft import fft
from fypy.pricing.StrikesPricer import StrikesPricer


class CarrMadanEuropeanPricer(StrikesPricer):
    def __init__(
        self, model: FourierModel, alpha: float = 0.75, eta: float = 0.1, N: int = 2**9
    ):
        """Carr-Madan method for Pricing European options under a Fourier model (i.e. using ChF)


        Args:
            model (FourierModel): FourierModel, model to price under
            alpha (float, optional):  Defaults to 0.75.
            eta (float, optional):  Defaults to 0.1.
            N (int, optional):  Defaults to 2**9.
        """
        self._model = model
        self._alpha = alpha
        self._eta = eta
        self._N = N
        self._logS0 = np.log(self._model.spot())

    def price(self, T: float, K: float, is_call: bool) -> float:
        """
        Price a single strike of European option
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        lam = 2 * np.pi / (self._N * self._eta)
        b = self._N * lam / 2

        uv = np.arange(1, self._N + 1)  # TODO: check
        ku = -b + lam * (uv - 1)
        vj = (uv - 1) * self._eta

        psij = self._chf(T=T, xi=vj - (self._alpha + 1) * 1j) / (
            self._alpha**2 + self._alpha - vj**2 + 1j * (2 * self._alpha + 1) * vj
        )

        disc = self._model.discountCurve(T)

        temp = (disc * self._eta / 3) * np.exp(1j * vj * b) * psij
        ind = np.zeros_like(uv)
        ind[0] = 1
        temp = temp * (3 + (-1) ** uv - ind)

        Cku = np.real(np.exp(-self._alpha * ku) * fft(temp) / np.pi)

        logK = np.log(K)
        istrike = int(np.floor((logK + b) / lam + 1)) - 1

        xp = [ku[istrike], ku[istrike + 1]]
        yp = [Cku[istrike], Cku[istrike + 1]]
        price = float(np.interp(logK, xp, yp))
        if not is_call:
            price = price - float((self._model.forwardCurve(T) * disc - K * disc))
        return price

    def _chf(self, T: float, xi: np.ndarray):
        return self._model.chf(T, xi) * np.exp(1j * self._logS0 * xi)
