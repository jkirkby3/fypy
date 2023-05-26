import unittest

from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.model.levy.VarianceGamma import VarianceGamma
from fypy.model.levy.NIG import NIG
from fypy.model.levy.CGMY import CMGY
from fypy.model.levy.MertonJD import MertonJD
from fypy.model.levy.KouJD import KouJD
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
from fypy.pricing.analytical.black_scholes import black76_price, black76_price_strikes


class Test_Proj_European(unittest.TestCase):
    def test_price_black_scholes(self):
        S0 = 100
        r = 0.01
        q = 0.03
        T = 1
        sigma = 0.2

        order = 3

        N = 2**10 if order==3 else 2**13
        precision= 13 if order==3 else 11

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
        model = BlackScholes(sigma=sigma, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, order=order)

        # Sanity Check the term structures
        self.assertAlmostEqual(fwd(T), S0 * div_disc(T) / disc_curve(T), precision)
        self.assertAlmostEqual(fwd.drift(0, T), (r - q) * T, precision)

        for is_call in [True]:
            for K in [S0 - 10, S0, S0 + 10]:
                price = pricer.price(T=T, K=K, is_call=is_call)
                true_price = black76_price(F=fwd(T), K=K, is_call=is_call, vol=sigma, disc=disc_curve(T), T=T)
                self.assertAlmostEqual(price, true_price, precision)

        strikes = np.asarray([S0-25, S0, S0+5, S0+25])
        is_calls = np.asarray([True for _ in range(len(strikes))])
        prices_bsm = black76_price_strikes(F=fwd(T), K=strikes, is_calls=is_calls, vol=sigma, disc=disc_curve(T), T=T)
        prices_proj = pricer.price_strikes(T=T, K=strikes, is_calls=is_calls)

        for i in range(len(prices_bsm)):
            self.assertAlmostEqual(prices_bsm[i], prices_proj[i], precision)

    def test_price_match_levy(self):
        S0 = 100
        r = 0.05
        q = 0.01
        T = 1

        order = 3

        N = 2 ** 14
        precision = 13 if order == 3 else 11

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

        # 1) Black Scholes
        model = BlackScholes(sigma=0.15, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, L=12, order=order)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 7.94871378854164, precision)

        # 2) Variance Gamma
        model = VarianceGamma(sigma=0.2, theta=0.1, nu=0.85, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, L=12, order=order)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 10.13935062748614, precision)

        # 3) NIG
        model = NIG(alpha=15, beta=-5, delta=0.5, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, L=12, order=order)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 9.63000693130414, precision)

        # 4) MJD
        model = MertonJD(sigma=0.12, lam=0.4, muj=-0.12, sigj=0.18, forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, L=12, order=order)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 8.675684635426279, precision)

        # 5) CGMY
        model = CMGY(forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, L=12, order=order)
        price = pricer.price(T=T, K=S0, is_call=True)
        self.assertAlmostEqual(price, 5.80222163947386, precision)

        # 6) Kou's Jump Diffusion
        model = KouJD(forwardCurve=fwd, discountCurve=disc_curve)
        pricer = ProjEuropeanPricer(model=model, N=N, L=12, order=order)
        price = pricer.price(T=T, K=S0, is_call=True)

        self.assertAlmostEqual(price, 11.92430307601936, precision)


if __name__ == '__main__':
    unittest.main()
