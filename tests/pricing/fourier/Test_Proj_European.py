import unittest

from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.model.levy.VarianceGamma import VarianceGamma
from fypy.model.levy.NIG import NIG
from fypy.model.levy.CGMY import CMGY
from fypy.model.levy.MertonJD import MertonJD
from fypy.model.levy.KouJD import KouJD
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

    def test_price_match_levy(self):
        S0 = 100
        r = 0.05
        q = 0.01
        T = 1

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

        # 1) Black Scholes
        model = BlackScholes(sigma=0.15, forwardCurve=fwd)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 14, L=12)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 7.94871378854164, 13)

        # 2) Variance Gamma
        model = VarianceGamma(sigma=0.2, theta=0.1, nu=0.85, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 14, L=12)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 10.13935062748614, 13)

        # 3) NIG
        model = NIG(alpha=15, beta=-5, delta=0.5, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 14, L=12)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 9.63000693130414, 13)

        # 4) MJD
        model = MertonJD(sigma=0.12, lam=0.4, muj=-0.12, sigj=0.18, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 14, L=12)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 8.675684635426279, 13)

        # 5) CGMY
        model = CMGY(forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 14, L=12)
        price = pricer.price(T=T, K=S0, is_call=True)
        self.assertAlmostEqual(price, 5.80222163947386, 13)

        # 6) Kou's Jump Diffusion
        model = KouJD(forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=2 ** 14, L=12)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 11.92430307601936, 13)


if __name__ == '__main__':
    unittest.main()
