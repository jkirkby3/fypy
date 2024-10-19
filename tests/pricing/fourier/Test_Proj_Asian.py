import os
import unittest
from unittest.mock import patch

from scipy.io import loadmat as loadmat

from fypy.model.levy.BilateralGamma import BilateralGammaMotion, BilateralGamma
from fypy.model.levy.BlackScholes import *
from fypy.pricing.fourier.ProjAsianPricer import ProjArithmeticAsianPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward


class Test_Proj_Arithmetic_Asian(unittest.TestCase):
    def test_arithmetic_asian_bilateral_gamma_motion(self):
        # Load of MATLAB results
        # Get the absolute path to the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the .mat file
        file_path = os.path.join(script_dir, '..', '..', 'numerical_values_for_testing',
                                 'prices_arithmetic_asian_mat.mat')

        # Load the .mat file
        matlab_prices = loadmat(file_path)['prices']

        original_cumulants = BilateralGamma.cumulants

        def patched_cumulants(self, T: float):
            cumulants = original_cumulants(self, T)
            cumulants.c2 += self.sigma**2
            return cumulants

        with patch.object(BilateralGammaMotion, 'cumulants', new=patched_cumulants):
            # Model and Pricer creation
            S0 = 100  # Initial price
            r = 0.05
            q = 0.02
            N = 2 ** 8
            disc_curve = DiscountCurve_ConstRate(rate=r)
            div_disc = DiscountCurve_ConstRate(rate=q)
            fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
            model = BilateralGammaMotion(forwardCurve=fwd, discountCurve=disc_curve, alpha_p=2.7884,
                                         lambda_p=12.2805, alhpa_m=0.2699, lambda_m=1.2018,
                                         sigma=0.4373)  # CHOOSE PROJ PARAMETERS
            pricer = ProjArithmeticAsianPricer(model=model, N=N)

            # Parameters
            put_call = np.asarray([False, True])
            T = np.arange(0.1, 2.0, 0.3)  # Time (in years)
            K = np.arange(40, 191, 2)  # Strike
            M = np.arange(48, 64, 4)  # Number of monitoring points

            # Testing multi-strike method
            for p_c in put_call:
                for t in T:
                    for m in M:
                        prices = pricer.price_strikes(T=t,
                                                      M=m,
                                                      K=K,
                                                      is_calls=np.full(len(K), p_c))

                        for k in range(len(K)):
                            self.assertAlmostEqual(prices[k], matlab_prices[np.where(put_call == p_c)[0][0],
                            k, np.where(T == t)[0][0], np.where(M == m)[0][0]], 10)

            # Testing single-strike method, selecting central elements
            inner_T = [T[len(T) // 2 - 1], T[len(T) // 2]] if len(T) % 2 == 0 else [T[len(T) // 2]]
            inner_K = [K[len(K) // 2 - 1], K[len(K) // 2]] if len(K) % 2 == 0 else [K[len(K) // 2]]

            for p_c in put_call:
                for t in inner_T:
                    for k in inner_K:
                        for m in M:
                            price = pricer.price(T=t, M=m, K=k, is_call=p_c)

                            self.assertAlmostEqual(price, matlab_prices[np.where(put_call == p_c)[0][0],
                            np.where(K == k)[0][0], np.where(T == t)[0][0], np.where(M == m)[0][0]], 3)

if __name__ == '__main__':
    unittest.main()
