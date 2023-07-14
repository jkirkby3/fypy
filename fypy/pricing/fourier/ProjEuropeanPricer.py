import numpy as np
from scipy.fft import fft

from fypy.model.levy.LevyModel import FourierModel
from fypy.pricing.fourier.ProjPricer import ProjPricer, Impl, CubicImpl, LinearImpl, HaarImpl


class ProjEuropeanPricer(ProjPricer):
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
        :param alpha_override: float, if supplied, this overrides the rule using L to determine the gridwidth,
            allows you to use your own rule to set grid if desired
        """
        self._model = model
        self._order = order
        self._N = N
        self._L = L
        self._alpha_override = alpha_override
        self._efficient_multi_strike = [1]

        if order not in (0, 1, 3):
            raise NotImplementedError("Only cubic, linear and Haar implemented so far")

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
        max_lws = np.log(np.max(K) / S0)

        cumulants = self._model.cumulants(T)
        alph = cumulants.get_truncation_heuristic(L=self._L) \
            if np.isnan(self._alpha_override) else self._alpha_override

        # Ensure that grid is wide enough to cover the strike
        alph = max(alph, 1.15 * max(np.abs(lws_vec)) + cumulants.c1)

        dx = 2 * alph / (self._N - 1)
        a = 1. / dx
        lam = cumulants.c1 - (self._N / 2 - 1) * dx

        max_n_bar = self.get_nbar(a=a, lws=max_lws, lam=lam)

        if self._order == 0:
            impl = HaarImpl(N=self._N, dx=dx, model=self._model, T=T, max_n_bar=max_n_bar)

        elif self._order == 1:
            impl = LinearImpl(N=self._N, dx=dx, model=self._model, T=T, max_n_bar=max_n_bar)

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
            nbar = self.get_nbar(a=a, lws=lws, lam=lam)
            xmin = lws - (nbar - 1) * dx

            beta = ProjEuropeanPricer._beta_computation(impl=impl, xmin=xmin)
            coeffs = impl.coefficients(nbar=nbar, W=strike, S0=S0, xmin=xmin)

            # price the put
            price = cons3 * np.dot(beta[:len(coeffs)], coeffs)
            if is_calls[index]:  # price using put-call parity
                price += (fwd - strike) * disc

            output[index] = max(0, price)

        # Prices method adapted to multi-strike
        # with Quadrature Adjustment for Grid Misalignment
        def price_misaligned_grid(index, strike):
            closest_nbar = self.get_nbar(a=a, lws=lws_vec[index], lam=xmin)
            rho = lws_vec[index] - (xmin + (closest_nbar - 1) * dx)

            coeffs = impl.coefficients(nbar=closest_nbar, W=strike, S0=S0, xmin=xmin, rho=rho,
                                       misaligned_grid=True)

            # price the put
            price = cons3 * np.dot(beta[:len(coeffs)], coeffs)
            if is_calls[index]:  # price using put-call parity
                price += (fwd - strike) * disc

            output[index] = max(0, price)

        # Prices computation

        if len(K) > 1 and self._order in self._efficient_multi_strike:
            xmin = max_lws - (max_n_bar - 1) * dx
            beta = ProjEuropeanPricer._beta_computation(impl=impl, xmin=xmin)
            price_vectorized = np.vectorize(price_misaligned_grid)
            price_vectorized(np.arange(0, len(K)), K)
        else:
            price_vectorized = np.vectorize(price_aligned_grid)
            price_vectorized(np.arange(0, len(K)), K)

    @staticmethod
    def _beta_computation(impl: Impl = None, xmin: float = None):
        return np.real(fft(impl.integrand(xmin=xmin)))
