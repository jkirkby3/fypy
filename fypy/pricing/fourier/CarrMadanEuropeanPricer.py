from fypy.termstructures.EquityForward import EquityForward
from fypy.model.FourierModel import FourierModel
import numpy as np
from scipy.fft import fft
from fypy.pricing.StrikesPricer import StrikesPricer


class CarrMadanEuropeanPricer(StrikesPricer):
    def __init__(
        self, model: FourierModel, alpha: float = 0.75, eta: float = 0.1, N: int = 2**9
    ):
        self._model = model
        self._alpha = alpha  # contour shift param (see Lee for recommendations)
        self._eta = eta
        self._N = N
        self._logS0 = np.log(self._model.spot())

    def price(self, T: float, K: float, is_call: bool):
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
        price = np.interp(logK, xp, yp)
        if not is_call:
            price = price - self._model.forwardCurve(T) * disc - K * disc
        return price

    def _chf(self, T: float, xi: np.ndarray):
        return self._model.chf(T, xi) * np.exp(1j * self._logS0 * xi)
