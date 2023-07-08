import os
import unittest

from scipy.io import loadmat as loadmat

from fypy.model.levy.BilateralGamma import BilateralGammaMotion
from fypy.model.levy.BlackScholes import *
from fypy.pricing.fourier.ProjBarrierPricer import ProjBarrierPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward



class Test_Proj_Barrier(unittest.TestCase):
    def test_barrier_bilateral_gamma_motion(self):
        # Load of MATLAB results
        # Get the absolute path to the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the .mat file
        file_path = os.path.join(script_dir, '..', '..', 'numerical_values_for_testing', 'prices_barrier_mat.mat')

        # Load the .mat file
        matlab_prices = loadmat(file_path)['prices']

        # Parameters
        S0 = 100
        r = 0.05
        q = 0.02

        T = np.arange(0.1, 2, 0.3)  # Time (in years)
        K = np.arange(90, 147, 7)  # Strike
        M = np.arange(40, 65, 12)  # number of discrete monitoring points
        H = np.arange(70, 89, 9)  # barrier

        is_calls = np.empty(len(K))
        is_calls.fill(True)
        N = 2 ** 14

        # Model and Pricer creation
        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
        model = BilateralGammaMotion(forwardCurve=fwd, discountCurve=disc_curve, alpha_p=1.3737 * 10 ** (-11),
                                     lambda_p=90.6317, alhpa_m=0.4331, lambda_m=2.4510, sigma=0.2725)
        pricer = ProjBarrierPricer(model=model, N=N)

        # Testint multi-strike method
        for t in T:
            for m in M:
                for h in H:
                    prices = pricer.price_strikes(T=t,
                                                  M=m,
                                                  H=h,
                                                  down=1,
                                                  rebate=5,
                                                  K=K,
                                                  is_calls=is_calls)
                    for w in range(len(K)):
                        self.assertAlmostEqual(prices[w], matlab_prices[
                            np.where(T == t)[0][0], w, np.where(M == m)[0][0], np.where(H == h)[0][0]], 6)

        # Testing single-strike method
        for t in T:
            for w in K:
                for m in M:
                    for h in H:
                        single_price = pricer.price(T=t,
                                                    M=m,
                                                    H=h,
                                                    down=1,
                                                    rebate=5,
                                                    K=w,
                                                    is_call=is_calls[0])
                        self.assertAlmostEqual(single_price, matlab_prices[
                            np.where(T == t)[0][0], np.where(K == w)[0][0], np.where(M == m)[0][0], np.where(H == h)[0][
                                0]], 6)


if __name__ == '__main__':
    unittest.main()
