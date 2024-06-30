import numpy as np


from fypy.model.FourierModel import FourierModel
from fypy.pricing.fourier.StochVol.ProjBermudanPricer_SV import ProjBermudanPricer_SV


class ProjAmericanRichardson:
    def __init__(self, model: FourierModel, N: int = 2**11):
        self._bermudan_pricer = ProjBermudanPricer_SV(model, N)
        return

    def price(self, T: float, W: int, S0: float, M: int, is_call: bool) -> float:
        price2M = self._bermudan_pricer.price(T, W, S0, 2 * M, is_call)
        priceM = self._bermudan_pricer.price(T, W, S0, M, is_call)
        return 2 * price2M - priceM
