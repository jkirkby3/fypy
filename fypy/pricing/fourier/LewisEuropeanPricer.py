from fypy.model.FourierModel import FourierModel
import numpy as np
from scipy.fft import ifft
from fypy.pricing.StrikesPricer import StrikesPricer
from scipy.interpolate import interp1d
from scipy.integrate import quad


class LewisEuropeanPricer(StrikesPricer):
    def __init__(self,
                 model: FourierModel,
                 N: int = 2 ** 12,
                 interp: str = 'cubic'):
        """
        Price European options using Fourier method of Lewis (2001)
        :param model: Fourier model
        :param N: int (power of 2), number of quadrature points in integral calculation
        :param interp: str, 'cubic' or 'linear'
        """
        self._model = model
        self._limit = 200  # upper bound on number of quad subintervals
        self._N = N
        self._interp = interp

    def price_strikes_fill(self,
                           T: float,
                           K: np.ndarray,
                           is_calls: np.ndarray,
                           output: np.ndarray):
        """
        Price a set of set of strikes (at same time to maturity, ie one slice of a surface), using FFT with
        interpolation between strikes (note: using one by one pricing can be more accurate but much slower)
        :param T: float, time to maturity of options
        :param K: np.array, strikes of options
        :param is_calls: np.array[bool], indicators of if strikes are calls (true) or puts (false)
        :param output: np.ndarray[float], the output to fill in with prices, must be same size as K and is_calls
        :return: None, this method fills in the output array, make sure its sized properly first
        """
        N = self._N
        dx = self._limit / N
        x = np.arange(N) * dx  # the final value limit is excluded

        weight = np.arange(N)  # Simpson weights
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[N - 1] = 1

        dk = 2 * np.pi / self._limit
        b = N * dk / 2
        ks = -b + dk * np.arange(N)

        S0 = self._model.spot()
        chf = lambda x: self._model.chf(T=T, xi=x)

        integrand = np.exp(- 1j * b * np.arange(N) * dx) * chf(x - 0.5j) * 1 / (x ** 2 + 0.25) * weight * dx / 3
        integral_value = np.real(ifft(integrand) * N)
        disc = self._model.discountCurve(T)

        sf = self._model.forwardCurve(T) * disc

        if self._interp == "linear":
            spline = interp1d(ks, integral_value, kind='linear')
        elif self._interp == "cubic":
            spline = interp1d(ks, integral_value, kind='cubic')
        else:
            raise NotImplementedError("Only linear and cubic interpolation supported")

        output = sf - np.sqrt(S0 * K) * disc / np.pi * spline(np.log(S0 / K))

        output[~is_calls] -= sf - K[~is_calls] * disc

    def price(self, T: float, K: float, is_call: bool):
        """
        Price a single strike of European option using Quadrature implementation of Lewis
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        S0 = self._model.spot()
        cf = lambda x: self._model.chf(T=T, xi=x)
        disc = self._model.discountCurve(T)

        k = np.log(S0 / K)
        integrand = lambda u: np.real(np.exp(u * k * 1j) * cf(u - 0.5j)) * 1 / (u ** 2 + 0.25)
        int_value = quad(integrand, 0, self._N, limit=self._limit)[0]
        sf = self._model.forwardCurve(T) * disc
        price = sf - np.sqrt(S0 * K) * disc / np.pi * int_value
        if not is_call:
            price -= sf - K * disc

        return price
