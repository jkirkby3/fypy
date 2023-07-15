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

        # Model and Pricer creation
        S0 = 100
        r = 0.05
        q = 0.02
        N = 2 ** 14
        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
        model = BilateralGammaMotion(forwardCurve=fwd, discountCurve=disc_curve, alpha_p=1.3737 * 10 ** (-11),
                                     lambda_p=90.6317, alhpa_m=0.4331, lambda_m=2.4510, sigma=0.2725)
        pricer = ProjBarrierPricer(model=model, N=N)

        T = np.asarray([0.1, 0.7, 1, 1.9])  # Time (in years)
        K = np.arange(80, 201, 2)  # Strike
        M = np.asarray([40, 65])  # number of discrete monitoring points
        H = 70  # barrier
        rebate = 5
        is_calls = np.empty(len(K))
        is_calls.fill(True)

        # Testing multi-strike method
        for t in T:
            for m in M:
                prices = pricer.price_strikes(T=t,
                                              M=m,
                                              H=H,
                                              down=True,
                                              rebate=rebate,
                                              K=K,
                                              is_calls=is_calls)
                for k in range(len(K)):
                    self.assertAlmostEqual(prices[k], matlab_prices[
                        np.where(T == t)[0][0], k, np.where(M == m)[0][0]], 6)

        # Testing single-strike method, selecting central elements

        inner_T = [T[len(T) // 2 - 1], T[len(T) // 2]] if len(T) % 2 == 0 else [T[len(T) // 2]]
        inner_K = [K[len(K) // 2 - 1], K[len(K) // 2]] if len(K) % 2 == 0 else [K[len(K) // 2]]
        inner_M = [M[len(M) // 2 - 1], M[len(M) // 2]] if len(M) % 2 == 0 else [M[len(M) // 2]]

        for t in inner_T:
            for k in inner_K:
                for m in inner_M:
                    single_price = pricer.price(T=t,
                                                M=m,
                                                H=H,
                                                down=True,
                                                rebate=rebate,
                                                K=k,
                                                is_call=is_calls[0])

                    self.assertAlmostEqual(single_price, matlab_prices[
                        np.where(T == t)[0][0], np.where(K == k)[0][0], np.where(M == m)[0][0]], 6)


if __name__ == '__main__':
    unittest.main()
