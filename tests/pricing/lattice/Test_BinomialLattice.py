import unittest
import numpy as np

from fypy.pricing.lattice.BinomialLattice import binomial_lattice_black_scholes


class Test_BinomialLattice(unittest.TestCase):
    def test__basic(self):
        option = binomial_lattice_black_scholes(S_0 = 100,
                                                K = 100,
                                                r = 0.05,
                                                T = 1,
                                                sigma = 0.2,
                                                M = 10000,
                                                call = 1,
                                                is_american = True)

        self.assertAlmostEqual(option, 10.4503836)
      
