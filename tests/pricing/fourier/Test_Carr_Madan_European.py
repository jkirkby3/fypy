import unittest

from fypy.model.levy import VarianceGamma, NIG, CMGY, KouJD
from fypy.model.levy.MertonJD import MertonJD
from fypy.pricing.fourier.CarrMadanEuropeanPricer import CarrMadanEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward


class Test_Lewis_European(unittest.TestCase):

    def test_levy_models(self):
        S0 = 100
        r = 0.05
        q = 0.01
        T = 1

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

        # 1) Black Scholes
        model = BlackScholes(sigma=0.15, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = CarrMadanEuropeanPricer(model=model, N=2**20)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 7.94871378854164, 5)

        # 2) Variance Gamma
        model = VarianceGamma(
            sigma=0.2, theta=0.1, nu=0.85, forwardCurve=fwd, discountCurve=disc_curve
        )
        pricer = CarrMadanEuropeanPricer(model=model, N=2**20)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 10.13935062748614, 5)

        # 3) NIG
        model = NIG(
            alpha=15, beta=-5, delta=0.5, forwardCurve=fwd, discountCurve=disc_curve
        )
        pricer = CarrMadanEuropeanPricer(model=model, N=2**20)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 9.63000693130414, 5)

        # 4) MJD
        model = MertonJD(
            sigma=0.12,
            lam=0.4,
            muj=-0.12,
            sigj=0.18,
            forwardCurve=fwd,
            discountCurve=disc_curve,
        )
        pricer = CarrMadanEuropeanPricer(model=model, N=2**20)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 8.675684635426279, 5)

        # 5) CGMY
        model = CMGY(forwardCurve=fwd, discountCurve=disc_curve)
        pricer = CarrMadanEuropeanPricer(model=model, N=2**20)
        price = pricer.price(T=T, K=S0, is_call=True)
        self.assertAlmostEqual(price, 5.80222163947386, 5)

        # 6) Kou's Jump Diffusion
        model = KouJD(forwardCurve=fwd, discountCurve=disc_curve)
        pricer = CarrMadanEuropeanPricer(model=model, N=2**20)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 11.92430307601936, 5)


if __name__ == "__main__":
    unittest.main()
