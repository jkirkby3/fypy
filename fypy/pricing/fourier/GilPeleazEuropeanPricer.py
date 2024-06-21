from fypy.model.FourierModel import FourierModel
from fypy.pricing.StrikesPricer import StrikesPricer

import numpy as np
from scipy.integrate import quad


class GilPeleazEuropeanPricer(StrikesPricer):
    def __init__(self,
                 model: FourierModel,
                 limit: int = 1000):
        """
        Gil-Peleaz method for Pricing European options under a Fourier model (i.e. using ChF)
        :param model: FourierModel, model to price under
        :param limit: float, integration limit for Fourier integrals
        """
        self._model = model
        self._limit = int(limit)  # integration limit for Fourier integrals

    def price(self, T: float, K: float, is_call: bool):
        """
        Price a single strike of European option
        :param T: float, time to maturity
        :param K: float, strike of option
        :param is_call: bool, indicator of if strike is call (true) or put (false)
        :return: float, price of option
        """
        S0 = self._model.spot()
        k = np.log(K / S0)
        chf = lambda x: self._model.chf(T=T, xi=x)

        integrand1 = lambda u: np.real((np.exp(-u * k * 1j) / (u * 1j)) * chf(u - 1j) / chf(-1j))
        int1 = 1 / 2 + 1 / np.pi * quad(integrand1, 1e-15, np.inf, limit=self._limit)[0]

        integrand2 = lambda u: np.real(np.exp(-u * k * 1j) / (u * 1j) * chf(u))
        int2 = 1 / 2 + 1 / np.pi * quad(integrand2, 1e-15, np.inf, limit=self._limit)[0]

        disc = self._model.discountCurve(T)
        sf = self._model.forwardCurve(T) * disc
        price = sf * int1 - K * disc * int2

        if not is_call:
            price -= sf - K * disc

        return price