import unittest

from fypy.pricing.fourier.StochVol.ProjBermudanPricer_SV import ProjBermudanPricer_SV
from Test_StochVol_Helper import GenericTest


class Test_Proj_Bermudan(unittest.TestCase, GenericTest):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        GenericTest.__init__(self, option_name="Bermudan")

    def check_equality(self, price, matlab_price):
        self.assertAlmostEqual(price, matlab_price, 7)

    def _get_price(self, list_index):
        idx_W, idx_T, idx_M = list_index
        price = self.pricer.price(
            T=self.matlab.T[idx_T],
            M=self.matlab.M[idx_M],
            W=self.matlab.W[idx_W],
            S0=self.mkt.S0,
            is_call=False,
        )
        return price

    def _set_option_constants(self):
        self.contract = 1
        return

    def _set_pricer(self):
        self.pricer = ProjBermudanPricer_SV(model=self.model, N=self.pricer_params.N)


if __name__ == "__main__":
    unittest.main()
