import unittest

from fypy.pricing.analytical.black_scholes import black76_price
from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76


class Test_ImpliedVol_Black76(unittest.TestCase):
    def test_iv_b76(self):
        from fypy.termstructures.EquityForward import EquityForward
        from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate

        spot = 100
        disc_curve = DiscountCurve_ConstRate(rate=0.0)
        fwd_curve = EquityForward(S0=spot, discount=disc_curve)
        ivc = ImpliedVolCalculator_Black76(fwd_curve=fwd_curve, disc_curve=disc_curve)

        p = 0.05
        K_ = 100
        T = 0.5
        v = ivc.imply_vol(price=p, strike=K_, ttm=T, is_call=True)
        p2 = black76_price(fwd_curve(T), K_, True, v, disc_curve(T), T)

        self.assertAlmostEqual(p2, p, 14)
