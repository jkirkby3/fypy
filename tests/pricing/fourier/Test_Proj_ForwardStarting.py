import os
import unittest

from scipy.io import loadmat as loadmat


from fypy.model.levy.BlackScholes import *
from fypy.model.levy.MertonJD import MertonJD

from fypy.pricing.fourier.ProjForwardStarting import ProjForwardStartingOption
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward


class Test_Proj_ForwardStarting(unittest.TestCase):
    def test_forward_MJD(self):

        # N.B. Regarding alphas computation, check _TODO_ comment in ProjBarrier

        # Load of MATLAB results
        # Get the absolute path to the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the .mat file
        file_path = os.path.join(
            script_dir,
            "tests",
            # "pricing",
            "numerical_values_for_testing",
            "prices_startingForward.mat",
        )

        # Load the .mat file
        matlab_prices = loadmat(file_path)  # ["prices"]
        # Model and Pricer creation
        S0 = matlab_prices["S_0"][0, 0]
        r = matlab_prices["r"][0, 0]
        q = matlab_prices["q"][0, 0]
        N = 2**14
        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)
        model = MertonJD(fwd, disc_curve, sigma=0.12, lam=0.4, muj=-0.12, sigj=0.18)
        pricer = ProjForwardStartingOption(model=model, N=N)

        start_dates = matlab_prices["lin_tstar"][0]
        taus = matlab_prices["lin_tau"][0]
        print(f"Number of start dates tested : {len(start_dates)}.")
        print(f"Number of expiry per start date : {len(taus)}.")

        # Testing multi-strike method
        errors = []
        for start_date_idx in range(len(start_dates)):
            for tau_idx in range(len(taus)):
                start_date = start_dates[start_date_idx]
                tau = taus[tau_idx]
                prices = pricer.price_strikes_fill(
                    start_date=start_date, tau=tau, S_0=1
                )
                # for k in range(len(K)):
                self.assertAlmostEqual(
                    prices,
                    matlab_prices["prices"][start_date_idx][tau_idx],
                    6,
                )
                errors.append(
                    np.abs(prices - matlab_prices["prices"][start_date_idx][tau_idx])
                )

        print(f"Error L^inf: {np.max(errors)}.")


if __name__ == "__main__":
    unittest.main()
