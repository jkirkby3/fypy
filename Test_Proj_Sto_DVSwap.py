import unittest


from fypy.pricing.fourier.StochVol.ProjDVSwap_SV import ProjDVSwap_SV

from Test_StochVol_Helper import GenericTest


class Test_Proj_DVSwap(unittest.TestCase, GenericTest):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        GenericTest.__init__(self, option_name="DVSwap")

    def check_equality(self, price, matlab_price):
        self.assertAlmostEqual(price, matlab_price, 7)

    def _get_price(self, list_index):
        idx_W, idx_T, idx_M = list_index
        price = self.pricer.price(
            T=self.matlab.T[idx_T],
            M=self.matlab.M[idx_M],
            W=self.matlab.W[idx_W],
            contract=self.contract,
        )
        return price

    def _set_option_constants(self):
        self.contract = 1
        return

    def _set_pricer(self):
        self.pricer = ProjDVSwap_SV(model=self.model, N=self.pricer_params.N, L=14)


if __name__ == "__main__":
    unittest.main()
