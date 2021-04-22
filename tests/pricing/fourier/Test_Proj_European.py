import unittest

from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.pricing.analytical.black_scholes import black76_price


class Test_Proj_European(unittest.TestCase):
    def test_price_black_scholes(self):
        S0 = 100
        r = 0.01
        q = 0.03
        T = 1
        sigma = 0.2

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
        model = BlackScholes(sigma=sigma, forwardCurve=fwd)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 10)

        # Sanity Check the term structures
        self.assertAlmostEqual(fwd(T), S0 * div_disc(T) / disc_curve(T), 13)
        self.assertAlmostEqual(fwd.drift(0, T), (r - q) * T, 13)

        for is_call in [True, False]:
            for K in [S0 - 10, S0, S0 + 10]:
                price = pricer.price(T=T, K=K, is_call=is_call)
                true_price = black76_price(F=fwd(T), K=K, is_call=is_call, vol=sigma, disc=disc_curve(T), T=T)
                self.assertAlmostEqual(price, true_price, 13)


if __name__ == '__main__':
    unittest.main()
