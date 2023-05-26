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
        self._efficient_multi_strike= [1]

        if order not in (0, 1, 3):
            raise NotImplementedError("Only cubic and Haar implemented so far")

    def price_strikes_fill(self,
                           T: float,
                           K: np.ndarray,
                           is_calls: np.ndarray,
                           output: np.ndarray):
        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface)
        Override this method if given a more efficient implementation for multiple strikes.

        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param is_calls: np.ndarray[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :return: None, this method fills in the output array, make sure its sized properly first
        """
        S0 = self._model.spot()
        lws_vec = np.log(K / S0)

        cumulants = self._model.cumulants(T)
        alph = cumulants.get_truncation_heuristic(L=self._L) \
            if np.isnan(self._alpha_override) else self._alpha_override

        # Ensure that grid is wide enough to cover the strike
        alph = max(alph, 1.15 * max(np.abs(lws_vec)) + cumulants.c1)

        dx = 2 * alph / (self._N - 1)
        a = 1. / dx
        lam = cumulants.c1 - (self._N / 2 - 1) * dx

        max_lws = np.log(np.max(K) / S0)
        max_n_bar = self._get_nbar(a=a, lws=max_lws, lam=lam)


        if self._order==0:
            impl= HaarImpl(N=self._N, dx=dx, model=self._model, T=T, max_n_bar=max_n_bar)

        elif self._order==1:
            impl= LinearImpl(N=self._N, dx=dx, model=self._model, T=T, max_n_bar=max_n_bar)

        else:
            impl = CubicImpl(N=self._N, dx=dx, model=self._model, T=T, max_n_bar=max_n_bar)

        disc = self._model.discountCurve(T)
        fwd = self._model.forwardCurve(T)
        cons3 = impl.cons() * disc / self._N

        # ==============
        # Price Strikes
        # ==============


        def price_aligned_grid(index, strike):
            lws = lws_vec[index]
            nbar = self._get_nbar(a=a, lws=lws, lam=lam)
            xmin = lws - (nbar - 1) * dx

            beta = np.real(fft(impl.integrand(xmin=xmin)))
            coeffs = impl.coefficients(nbar=nbar, W=strike, S0=S0, xmin=xmin)

            # price the put
            price = cons3 * np.dot(beta[:len(coeffs)], coeffs)
            if is_calls[index]:  # price using put-call parity
                price += (fwd - strike) * disc

            output[index] = max(0, price)

        # Prices method adapted to multi-strike
        # with Quadrature Adjustment for Grid Misalignment
        def price_misaligned_grid(index, strike):
            closest_nbar = self._get_nbar(a=a, lws=lws_vec[index], lam=xmin)
            rho = lws_vec[index] - (xmin + (closest_nbar - 1) * dx)

            coeffs = impl.coefficients(nbar=closest_nbar, W=strike, S0=S0, xmin=xmin, rho=rho,
                                                    misaligned_grid=True)

            # price the put
            price = cons3 * np.dot(beta[:len(coeffs)], coeffs)
            if is_calls[index]:  # price using put-call parity
                price += (fwd - strike) * disc

            output[index] = max(0, price)


        # Prices computation

        if len(K)>1 and self._order in self._efficient_multi_strike:
            xmin = max_lws - (max_n_bar - 1) * dx
            beta = np.real(fft(impl.integrand(xmin=xmin)))
            price_vectorized = np.vectorize(price_misaligned_grid)
            price_vectorized(np.arange(0, len(K)), K)


        else:
            price_vectorized = np.vectorize(price_aligned_grid)
            price_vectorized(np.arange(0,len(K)),K)



    def _get_nbar(self, a: float, lws: float, lam: float) -> int:
        try:
            nbar = int(np.floor(a * (lws - lam) + 1))
            if nbar >= self._N:
                nbar = self._N - 1
        except Exception as e:
            raise e
        return nbar


# ===================================
# Private
# ===================================


class Impl(ABC):
    """ Implementation of the PROJ method. Extended by Cubic, Quadratic, Linear, Haar """

    def __init__(self,
                 N: int,
                 dx: float,
                 model: FourierModel,
                 T,
                 max_n_bar: int):
        self.N = N
        self.dx = dx
        self.a = 1. / dx
        self.model = model
        self.T = T
        self.G = np.zeros(self.num_coeffs(nbar=max_n_bar))  # payoff coefficients

        dw = 2 * np.pi / (N * dx)
        self.w = dw * np.arange(0, N)

    @abstractmethod
    def integrand(self, xmin: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def coefficients(self, nbar: int, W: float, S0: float, xmin: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cons(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def num_coeffs(self, nbar: int) -> int:
        raise NotImplementedError


# ==============
# Cubic Basis
# ==============
class CubicImpl(Impl):
    def __init__(self,
                 N: int,
                 dx: float,
                 model: FourierModel,
                 T,
                 max_n_bar: int):
        super(CubicImpl, self).__init__(N=N, dx=dx, model=model, T=T, max_n_bar=max_n_bar)
        self._init_consts()

        # precompute components of the fourier integrand that dont depend on strike
        self._base_integrand = self._init_base_integrand()

        # Precompute the exponentials needed to evaluate payoff
        self._expos = np.exp(self.dx * np.arange(0, max_n_bar - 2))

    def integrand(self, xmin: float) -> np.ndarray:
        return self._base_integrand * np.exp(-1j * xmin * self.w)

    def num_coeffs(self, nbar: int) -> int:
        return nbar + 1

    def coefficients(self,
                     nbar: int, W: float, S0: float, xmin: float) -> np.ndarray:
        self.G[nbar] = W * self.g1
        self.G[nbar - 1] = W * self.g2
        self.G[nbar - 2] = W * self.g3
        # TODO: for multi strike case, can reuse the expos
        self.G[: nbar - 2] = W - S0 * np.exp(xmin) * self._expos[:nbar - 2] / 90 * self.g4

        return self.G

    def cons(self):
        return 32 * self.a ** 4

    def _init_consts(self):
        dx = self.dx
        self.g1 = (1 / 24 - 1 / 20 * np.exp(dx)
                   * (np.exp(-7 / 4 * dx) / 54 + np.exp(-1.5 * dx) / 18
                      + np.exp(-1.25 * dx) / 2 + 7 * np.exp(-dx) / 27))

        self.g2 = (.5 - .05 * (28 / 27 + np.exp(-7 / 4 * dx) / 54
                               + np.exp(-1.5 * dx) / 18 + np.exp(-1.25 * dx) / 2
                               + 14 * np.exp(-dx) / 27 + 121 / 54 * np.exp(-.75 * dx)
                               + 23 / 18 * np.exp(-.5 * dx) + 235 / 54 * np.exp(-.25 * dx)))

        self.g3 = (23 / 24 - np.exp(-dx) / 90
                   * ((28 + 7 * np.exp(-dx)) / 3 +
                      (14 * np.exp(dx) + np.exp(-7 / 4 * dx)
                       + 242 * np.cosh(.75 * dx) + 470 * np.cosh(.25 * dx)) / 12
                      + .25 * (np.exp(-1.5 * dx) + 9 * np.exp(-1.25 * dx) + 46 * np.cosh(.5 * dx))))

        self.g4 = (14 / 3 * (2 + np.cosh(dx)) + .5 * (np.cosh(1.5 * dx) + 9 * np.cosh(1.25 * dx)
                                                      + 23 * np.cosh(.5 * dx)) + 1 / 6
                   * (np.cosh(7 / 4 * dx) + 121 * np.cosh(.75 * dx) + 235 * np.cosh(.25 * dx)))

    def _init_base_integrand(self) -> np.ndarray:
        b0 = 1208 / 2520
        b1 = 1191 / 2520
        b2 = 120 / 2520
        b3 = 1 / 2520

        a = self.a
        w = self.w[1:]
        grand = np.empty_like(self.w, dtype=complex)

        grand[1:] = self.model.chf(T=self.T, xi=w) \
                * (np.sin(w / (2 * a)) / w) ** 4 \
                / (b0 + b1 * np.cos(w / a) + b2 * np.cos(2 * w / a) + b3 * np.cos(3 * w / a))

        grand[0] = 1 / self.cons()
        return grand

class LinearImpl(Impl):
    def __init__(self,
                 N: int,
                 dx: float,
                 model: FourierModel,
                 T,
                 max_n_bar: int):
        super(LinearImpl, self).__init__(N=N, dx=dx, model=model, T=T, max_n_bar=max_n_bar)
        self._init_consts()

        # precompute components of the fourier integrand that don't depend on strike
        self._base_integrand = self._init_base_integrand()

        # Precompute the exponentials needed to evaluate payoff
        self._expos = np.exp(self.dx * np.arange(0, max_n_bar - 1))

    def cons(self):
        return 24 * self.a ** 2

    def num_coeffs(self, nbar: int) -> int:
        return nbar + 1

    def _init_base_integrand(self) -> np.ndarray:
        a = self.a
        w = self.w[1:]
        grand = np.empty_like(self.w, dtype=complex)

        grand[1:] = (self.model.chf(T=self.T, xi=w) * (((np.sin(w / (2 * a))) / w) ** 2)) / (2 + np.cos(w / a))

        grand[0] = 1 / self.cons()
        return grand

    def integrand(self, xmin: float) -> np.ndarray:
        return self._base_integrand * np.exp(-1j * xmin * self.w)

    def _init_consts(self):
        dx = self.dx
        self.g1 = 0.5 - (1 / 15) * (
                7 / 6
                + 4 / 3 * np.exp(-3 / 4 * dx)
                + np.exp(-0.5 * dx)
                + 4 * np.exp(-0.25 * dx)
        )

        self.g2 = (
                          7 / 3 + 8 / 3 * np.cosh(3 / 4 * dx) + 2 * np.cosh(0.5 * dx) + 8 * np.cosh(0.25 * dx)
                  ) / 15

    def coefficients(self,
                     nbar: int, W: float, S0: float, xmin: float, rho: float = 0,
                     misaligned_grid: bool = False) -> np.ndarray:

        self.G[nbar - 1] = W * self.g1
        self.G[: nbar - 1] = W - S0 * np.exp(xmin) * self._expos[:nbar - 1] * self.g2

        if misaligned_grid == True:

            dx = self.dx
            zeta = self.a * rho

            qPlus = 0.5 * (1 + np.sqrt(3 / 5))
            qMinus = 0.5 * (1 - np.sqrt(3 / 5))

            zetaPlus = zeta * qPlus
            zetaMinus = zeta * qMinus

            rhoPlus = rho * qPlus
            rhoMinus = rho * qMinus


            # theta computation

            theta0 = (1 / 15) * (
                    7 / 6
                    + 4 / 3 * np.exp(-3 / 4 * dx)
                    + np.exp(-0.5 * dx)
                    + 4 * np.exp(-0.25 * dx)
            )

            # delta bar computation

            delta_bar_0 = zeta * (1 - 0.5 * zeta)

            delta_bar_P1 = zeta - delta_bar_0

            # delta computation

            delta0 = (zeta / 18) \
                     * (4 * (2 - zeta) * np.exp(rho / 2) \
                        + 5 \
                        * ((1 - zetaMinus) * np.exp(rhoMinus) + (1 - zetaPlus) * np.exp(rhoPlus))
                        )

            deltaP1 = (zeta / 18) * np.exp(-dx) \
                      * (
                              4 * zeta * np.exp(0.5 * rho) + 5 * (
                              zetaMinus * np.exp(rhoMinus) + zetaPlus * np.exp(rhoPlus))
                      )

            self.G[nbar] = W * (delta_bar_P1 - np.exp(-rho) * np.exp(dx) * deltaP1)

            self.G[nbar - 1] +=  W * (delta_bar_0 - np.exp(-rho) * (theta0 + delta0) + theta0 )



        return self.G

class HaarImpl(Impl):

    def cons(self):
        return 4 * self.a

    def num_coeffs(self, nbar: int) -> int:
        return nbar

    def integrand(self, xmin: float) -> np.ndarray:
        w = self.w
        model = self.model
        T = self.T

        grand = model.chf(T=T, xi=w) * np.exp(-1j * xmin * w) * (np.sin(w / (2 * self.a)) / w)
        grand[0] = 1 / self.cons()
        return grand

    def coefficients(self, nbar: int, W: float, S0: float, xmin: float) -> np.ndarray:
        a = self.a
        dx = 1 / a
        self.G[nbar - 1] = W * (.5 - a * (1 - np.exp(-.5 * dx)))
        self.G[: nbar - 1] = W - np.exp(xmin + dx * np.arange(0, nbar - 1)) * S0 * 2 * a * np.sinh(dx / 2)
        return self.G
