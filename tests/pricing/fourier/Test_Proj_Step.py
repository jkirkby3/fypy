import gc
import os
import unittest

import torch
from scipy.io import loadmat as loadmat

from fypy.model.levy.BilateralGamma import BilateralGammaMotion
from fypy.model.levy.BlackScholes import *
from fypy.pricing.fourier.ProjStepPricer import ProjStepPricer
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward


class Test_Proj_Step(unittest.TestCase):
    def test_step_bilateral_gamma_motion(self):
        gc.collect()

        # Load of MATLAB results
        # Get the absolute path to the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the .mat file
        file_path = os.path.join(script_dir, '..', '..', 'numerical_values_for_testing', 'prices_step_mat.mat')

        # Load the .mat file
        matlab_prices = loadmat(file_path)['prices']

        # Initial parameters
        cuda = False
        S_0 = 2000  # Initial price
        r = 0.05  # Interest rate
        q = 0.02
        N = 2 ** 10

        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S_0, discount=disc_curve, divDiscount=div_disc)
        model = BilateralGammaMotion(forwardCurve=fwd, discountCurve=disc_curve, alpha_p=2.7884,
                                     lambda_p=12.2805,
                                     alhpa_m=0.2699, lambda_m=1.2018,
                                     sigma=0.4373)  # CHOOSE PROJ PARAMETERS

        pricer = ProjStepPricer(model=model, N=N, cuda=cuda)

        T = np.asarray([0.1, 2])  # Time (in years)
        stepRho = np.asarray([-1, 20])
        K = np.arange(1000, 3000, 100)  # Strike
        M = 52
        H = 3100
        is_calls = np.empty(len(K))
        is_calls.fill(True)
        down = False

        for t in T:
            for rho in stepRho:
                prices = pricer.price_strikes(T=t, M=M, H=H, down=down, K=K, stepRho=rho,
                                              is_calls=is_calls)
                for k in range(len(K)):
                    self.assertAlmostEqual(prices[k], matlab_prices[
                        np.where(T == t)[0][0], np.where(stepRho == rho)[0][0], k], 8)

                torch.cuda.empty_cache()

        mid_strike=K[len(K)//2]

        for t in T:
            for rho in stepRho:
                price = pricer.price(T=t, M=M, H=H, down=down, K=mid_strike, stepRho=rho,
                                              is_call=is_calls[len(is_calls)//2])

                self.assertAlmostEqual(price, matlab_prices[
                    np.where(T == t)[0][0], np.where(stepRho == rho)[0][0],
                    np.where(K == mid_strike)[0][0]], 8)


if __name__ == '__main__':
    unittest.main()
