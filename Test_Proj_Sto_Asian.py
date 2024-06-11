import unittest

from fypy.pricing.fourier.StochVol.ProjAsianPricer_SV import ProjAsianPricer_SV
from Test_StochVol_Helper import GenericTest


class Test_Proj_Asian(unittest.TestCase, GenericTest):
    def __init__(self, methodName="runTest"):
        GenericTest.__init__(self, option_name="Asian")
        unittest.TestCase.__init__(self, methodName)

    def check_equality(self, price, matlab_price):
        self.assertAlmostEqual(price, matlab_price, 7)

    def _get_price(self, list_index):
        idx_W, idx_T, idx_M = list_index
        price = self.pricer.price(
            T=self.matlab.T[idx_T],
            M=self.matlab.M[idx_M],
            W=self.matlab.W[idx_W],
            S0=self.mkt.S0,
            is_call=True,
        )
        return price

    def _set_option_constants(self):
        return

    def _set_pricer(self):
        self.pricer = ProjAsianPricer_SV(
            model=self.model, P=self.pricer_params.P, Pbar=self.pricer_params.Pbar
        )


if __name__ == "__main__":
    unittest.main()
