from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.model.levy.LevyModel import FourierModel
import numpy as np
from scipy.fft import fft
from abc import ABC, abstractmethod


class ProjEuropeanPricer(StrikesPricer):
    def __init__(self,
                 model: FourierModel,
                 N: int = 2 ** 9,
                 L: float = 10.,
                 order: int = 3,
                 alpha_override: float = np.nan):
        """
        Price European options using the Frame Projection (PROJ) method of Kirkby (2015)

        Ref: JL Kirkby, SIAM Journal on Financial Mathematics, 6 (1), 713-747

        :param model: Fourier model
        :param N: int (power of 2), number of basis coefficients (increase to increase accuracy)
        :param L: float, controls gridwidth of density. A value of L = 10~14 works well... For Black-Scholes,
            L = 6 is fine, for heavy tailed processes such as CGMY, may want a larger value to get very high accuracy
        :param order: int, the Spline order: 0 = Haar, 1 = Linear, 2 = Quadratic, 3 = Cubic
            Note: Cubic is preferred, the others are provided for research purposes. Only 1 and 3 are currently coded
        :param alpha_override: float, if supplied, this ovverrides the rule using L to determine the gridwidth,
            allows you to use your own rule to set grid if desired
        """
        self._model = model
        self._order = order
        self._N = N
        self._L = L
        self._alpha_override = alpha_override

        if order not in (0, 3):
            raise NotImplementedError("Only cubic and Haar implemented so far")

    def price(self, T: float, K: float, is_call: bool):
        """
        Price a single strike of European option
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        S0 = self._model.spot()
        lws = np.log(K / S0)

        cumulants = self._model.cumulants(T)
        alph = cumulants.get_truncation_heuristic(L=self._L) \
            if np.isnan(self._alpha_override) else self._alpha_override

        # Ensure that grid is wide enough to cover the strike
        alph = max(alph, 1.05 * abs(lws))

        dx = 2 * alph / (self._N - 1)
        a = 1. / dx

        lam = cumulants.c1 - (self._N / 2 - 1) * dx
        nbar = int(np.floor(a * (lws - lam) + 1))
        if nbar >= self._N:
            nbar = self._N - 1

        xmin = lws - (nbar - 1) * dx

        dw = 2 * np.pi / (self._N * dx)
        omega = dw * np.arange(0, self._N)

        # ==============
        # Cubic Basis
        # ==============
        # TODO: make impl contain all the info, construct it from the beginning
        impl = CubicImpl(a=a, xmin=xmin) if self._order == 3 else HaarImpl(a=a, xmin=xmin)
        grand = impl.integrand(model=self._model, T=T, w=omega)
        beta = np.real(fft(grand))

        coeffs = impl.coefficients(nbar=nbar, W=K, S0=S0)

        disc = self._model.discountCurve(T)
        # price the put
        price = impl.cons() * disc / self._N * np.dot(beta[:len(coeffs)], coeffs)
        if is_call:  # price using put-call parity
            price += (self._model.forwardCurve(T) - K) * disc

        return price


# ===================================
# Private
# ===================================

class Impl(ABC):
    """ Implementation of the PROJ method. Extended by Cubic, Quadratic, Linear, Haar """

    def __init__(self, a: float, xmin: float):
        self.a = a
        self.xmin = xmin

    @abstractmethod
    def integrand(self, model: FourierModel, T, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def coefficients(self, nbar: int, W: float, S0: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cons(self) -> float:
        raise NotImplementedError


class CubicImpl(Impl):
    def integrand(self, model: FourierModel, T, w: np.ndarray) -> np.ndarray:
        b0 = 1208 / 2520
        b1 = 1191 / 2520
        b2 = 120 / 2520
        b3 = 1 / 2520

        a = self.a
        xmin = self.xmin

        grand = model.chf(T=T, xi=w) \
                * np.exp(-1j * xmin * w) \
                * (np.sin(w / (2 * a)) / w) ** 4 \
                / (b0 + b1 * np.cos(w / a) + b2 * np.cos(2 * w / a) + b3 * np.cos(3 * w / a))

        grand[0] = 1 / (32 * a ** 4)
        return grand

    def coefficients(self, nbar: int, W: float, S0: float) -> np.ndarray:
        dx = 1 / self.a
        xmin = self.xmin
        G = np.zeros(nbar + 1)
        G[nbar] = W * (1 / 24 - 1 / 20 * np.exp(dx)
                       * (np.exp(-7 / 4 * dx) / 54 + np.exp(-1.5 * dx) / 18
                          + np.exp(-1.25 * dx) / 2 + 7 * np.exp(-dx) / 27))

        G[nbar - 1] = W * (.5 - .05 * (28 / 27 + np.exp(-7 / 4 * dx) / 54
                                       + np.exp(-1.5 * dx) / 18 + np.exp(-1.25 * dx) / 2
                                       + 14 * np.exp(-dx) / 27 + 121 / 54 * np.exp(-.75 * dx)
                                       + 23 / 18 * np.exp(-.5 * dx) + 235 / 54 * np.exp(-.25 * dx)))

        G[nbar - 2] = W * (23 / 24 - np.exp(-dx) / 90
                           * ((28 + 7 * np.exp(-dx)) / 3 +
                              (14 * np.exp(dx) + np.exp(-7 / 4 * dx)
                               + 242 * np.cosh(.75 * dx) + 470 * np.cosh(.25 * dx)) / 12
                              + .25 * (np.exp(-1.5 * dx) + 9 * np.exp(-1.25 * dx) + 46 * np.cosh(.5 * dx))))

        G[: nbar - 2] = W - S0 * np.exp(xmin + dx * np.arange(0, nbar - 2)) / 90 \
                        * (14 / 3 * (2 + np.cosh(dx)) + .5 * (np.cosh(1.5 * dx) + 9 * np.cosh(1.25 * dx)
                                                              + 23 * np.cosh(.5 * dx)) + 1 / 6
                           * (np.cosh(7 / 4 * dx) + 121 * np.cosh(.75 * dx) + 235 * np.cosh(.25 * dx)))

        return G

    def cons(self):
        return 32 * self.a ** 4


class HaarImpl(Impl):

    def cons(self):
        return 4 * self.a

    def integrand(self, model: FourierModel, T, w: np.ndarray) -> np.ndarray:
        grand = model.chf(T=T, xi=w) * np.exp(-1j * self.xmin * w) * (np.sin(w / (2 * self.a)) / w)
        grand[0] = 1 / (4 * self.a)
        return grand

    def coefficients(self, nbar: int, W: float, S0: float) -> np.ndarray:
        dx = 1 / self.a
        a = self.a
        xmin = self.xmin
        G = np.zeros(nbar)
        G[nbar - 1] = W * (.5 - a * (1 - np.exp(-.5 * dx)))
        G[: nbar - 1] = W - np.exp(xmin + dx * np.arange(0, nbar - 1)) * S0 * 2 * a * np.sinh(dx / 2)
        return G
