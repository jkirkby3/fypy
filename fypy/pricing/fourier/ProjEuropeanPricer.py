import numpy as np
from scipy.fft import fft

from fypy.model.levy.LevyModel import FourierModel
from fypy.pricing.fourier.ProjPricer import ProjPricer, Impl, CubicImpl, LinearImpl, HaarImpl

class ProjEuropeanPricer(ProjPricer):
    def __init__(self, model: FourierModel, N: int = 2 ** 9, L: float = 10., order: int = 3,
                 alpha_override: float = np.nan):
        super().__init__(model, N, L, order, alpha_override)
        self._efficient_multi_strike = [1]

        if order not in (0, 1, 3):
            raise NotImplementedError("Only cubic, linear and Haar implemented so far")

    def price_strikes_fill(self, T: float, K: np.ndarray, is_calls: np.ndarray, output: np.ndarray):
        """
        Price a set of strikes (at same time to maturity)
        """
        lws_vec = np.log(K / self._model.spot())
        max_lws = np.log(np.max(K) / self._model.spot())

        cumulants = self._model.cumulants(T)
        alph = cumulants.get_truncation_heuristic(L=self._L) if np.isnan(self._alpha_override) else self._alpha_override
        alph = max(alph, 1.15 * max(np.abs(lws_vec)) + cumulants.c1)

        grid = {
            'dx': 2 * alph / (self._N - 1),
            'a': 1. / (2 * alph / (self._N - 1)),
            'lam': cumulants.c1 - (self._N / 2 - 1) * (2 * alph / (self._N - 1)),
            'cons3': None,  # Verrà popolato successivamente
            'max_nbar': self.get_nbar(a=1. / (2 * alph / (self._N - 1)), lws=max_lws, lam=cumulants.c1 - (self._N / 2 - 1) * (2 * alph / (self._N - 1)))
        }

        impl = self._get_implementation(self._order, T, grid['max_nbar'], grid['dx'])

        grid['cons3'] = impl.cons() * self._model.discountCurve(T) / self._N

        option = {
            'disc': self._model.discountCurve(T),
            'fwd': self._model.forwardCurve(T),
            'lws_vec': lws_vec,
            'is_calls': is_calls,
            'max_lws': max_lws,
            'K': K
        }

        self.price_computation(grid, option, impl, output)

    def price_computation(self, grid: dict, option: dict, impl: Impl, output: np.ndarray):
        """
        Compute prices for multiple strikes, handling both aligned and misaligned grids.
        """

        if len(option['K']) > 1 and self._order in self._efficient_multi_strike:
            xmin = option['max_lws'] - (grid['max_nbar'] - 1) * grid['dx']
            option['beta'] = ProjEuropeanPricer._beta_computation(impl=impl, xmin=xmin)
            price_vectorized = np.vectorize(self.price_misaligned_grid, excluded=['grid', 'option', 'impl', 'output'])
        else:
            price_vectorized = np.vectorize(self.price_aligned_grid, excluded=['grid', 'option', 'impl', 'output'])

        price_vectorized(np.arange(0, len(option['K'])), option['K'], grid=grid, option=option, impl=impl, output=output)

    def price_aligned_grid(self, index, strike, grid: dict, option: dict, impl: Impl, output: np.ndarray):
        """
        Price computation for aligned grid.
        """
        lws = option['lws_vec'][index]
        nbar = self.get_nbar(a=grid['a'], lws=lws, lam=grid['lam'])
        xmin = lws - (nbar - 1) * grid['dx']
        beta = ProjEuropeanPricer._beta_computation(impl=impl, xmin=xmin)
        coeffs = impl.coefficients(nbar=nbar, W=strike, S0=self._model.spot(), xmin=xmin)
        price = grid['cons3'] * np.dot(beta[:len(coeffs)], coeffs)
        if option['is_calls'][index]:
            price += (option['fwd'] - strike) * option['disc']
        output[index] = max(0, price)

    def price_misaligned_grid(self, index, strike, grid: dict, option: dict, impl: Impl, output: np.ndarray):
        """
        Price computation for misaligned grid.
        """
        lws = option['lws_vec'][index]
        closest_nbar = self.get_nbar(a=grid['a'], lws=lws, lam=grid['lam'])
        xmin = lws - (closest_nbar - 1) * grid['dx']
        rho = lws - (xmin + (closest_nbar - 1) * grid['dx'])
        beta = ProjEuropeanPricer._beta_computation(impl=impl, xmin=xmin)
        coeffs = impl.coefficients(nbar=closest_nbar, W=strike, S0=self._model.spot(), xmin=xmin, rho=rho, misaligned_grid=True)
        price = grid['cons3'] * np.dot(beta[:len(coeffs)], coeffs)
        if option['is_calls'][index]:
            price += (option['fwd'] - strike) * option['disc']
        output[index] = max(0, price)


    @staticmethod
    def _beta_computation(impl: Impl = None, xmin: float = None):
        return np.real(fft(impl.integrand(xmin=xmin)))
