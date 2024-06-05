#######################################################
#######################################################
#######################################################
#     THIS IS THE CODE FOR AMERICAN BUT I INVERT IT   #
#     BC JUSTIN CONFUSED BOTH OF THEM                 #
#######################################################
#######################################################
#######################################################
import numpy as np

# np.set_printoptions(precision=4)

from fypy.model.FourierModel import FourierModel


from fypy.pricing.fourier.StochVol.ProjBermudanPricer_SV import ProjBermudanPricer_SV


class ProjAmericanRichardson:
    def __init__(
        self,
        model: FourierModel,
        N: int = 2**11,
        L: float = 10.0,
        order: int = 3,
        alpha_override: float = np.nan,
    ):
        self._bermudan_pricer = ProjBermudanPricer_SV(
            model, N, L, order, alpha_override
        )
        return

    def price(self, T: float, W: int, S0: float, M: float, is_call: bool) -> float:
        price2M = self._bermudan_pricer.price(T, W, S0, 2 * M, is_call)
        priceM = self._bermudan_pricer.price(T, W, S0, M, is_call)
        return 2 * price2M - priceM
