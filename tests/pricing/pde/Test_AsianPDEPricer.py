import unittest
import numpy as np

from fypy.pricing.pde.AsianPDEPricer import AsianPDEPricer, AsianOption


class Test_AsianPDEPricer(unittest.TestCase):
    def test__basic(self):
        option = AsianOption(strike=100,
                             is_call=True,
                             observation_times={"B": np.array([1.0])},
                             future_expiries={"B": 1.0},
                             weights={"B": 1.0})
        pricer = AsianPDEPricer(instrument=option,
                                q=lambda t: 0.,
                                mu=lambda t: 0.01,
                                sigma=lambda x, y: 0.2,
                                Ny=100)

        value = pricer.price()

