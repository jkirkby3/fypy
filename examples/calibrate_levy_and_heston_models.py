"""
This example shows how to calibrate a Levy model or Heston's stochastic volatility model (choose your favorite below).
We do the following:
    1) Create a synthetic market surface, priced using Variance Gamma with set of "true" parameters
    2) Create a calibrator and set the market prices as targets
    3) Calibrate the chosen model, starting from some initial guess
    4) Show that the calibration "discovers" the true market parameters
"""
import numpy as np
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy import *
from fypy.model.sv.Heston import Heston
from fypy.termstructures.EquityForward import EquityForward
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.market.MarketSurface import MarketSlice, MarketSurface
from fypy.fit.Targets import Targets
from fypy.fit.Calibrator import Calibrator, LeastSquares
import matplotlib.pyplot as plt

# ============================
# Set Common Parameters
# ============================
S0 = 100  # Initial stock price
r = 0.01  # Interest rate
q = 0.03  # Dividend yield

# ============================
# Set Term Structures
# ============================
disc_curve = DiscountCurve_ConstRate(rate=r)
div_disc = DiscountCurve_ConstRate(rate=q)
fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

# ============================
# Create Model (to generate a synthetic market to fit to)
# ============================
model_name = 'NIG'  # Choose this name, then override params if desired

models = {
    'VG': VarianceGamma(sigma=0.2, theta=-0.1, nu=0.6, forwardCurve=fwd, discountCurve=disc_curve),
    'BG': BilateralGamma(alpha_p=1.18, lambda_p=10.57, alhpa_m=1.44, lambda_m=5.57, forwardCurve=fwd,
                         discountCurve=disc_curve),
    'NIG': NIG(alpha=10, beta=-3, delta=0.4, forwardCurve=fwd, discountCurve=disc_curve),
    'CGMY': CMGY(C=0.05, G=4, M=10, Y=1.3, forwardCurve=fwd, discountCurve=disc_curve),
    'MJD': MertonJD(sigma=0.15, lam=0.3, muj=-0.2, sigj=0.3, forwardCurve=fwd, discountCurve=disc_curve),
    'KDE': KouJD(sigma=0.14, lam=2., p_up=0.3, eta1=20, eta2=15, forwardCurve=fwd, discountCurve=disc_curve),
    'BSM': BlackScholes(sigma=0.2, forwardCurve=fwd, discountCurve=disc_curve),
    'Hes': Heston(v_0=0.04, theta=0.04, kappa=0.1, sigma_v=0.5, rho=-0.5, forwardCurve=fwd, discountCurve=disc_curve)
}

# Create the pricers and attach to each model
model = models.get(model_name)

true_params = model.get_params()
pricer = ProjEuropeanPricer(model=model, N=2 ** 11, L=16 if model_name == 'Hes' else 12)

# Initialize market surface, fill it in with slices
surface = MarketSurface()

ttms = [0.1, 0.5, 1., 3.]  # tenors in surface
strikes = np.arange(50, 150, 5)  # same strikes for each tenor, for simplicity
is_calls = np.ones(len(strikes), dtype=bool)

target_prices = []

for ttm in ttms:
    prices = pricer.price_strikes(T=ttm, K=strikes, is_calls=is_calls)
    market_slice = MarketSlice(T=ttm, F=fwd(ttm), disc=div_disc(ttm), strikes=strikes,
                               is_calls=is_calls, mid_prices=prices)
    # Add the slice to surface
    surface.add_slice(ttm, market_slice)

    # push back the target prices to fit to
    target_prices.append(prices)

# Full set of market target prices
target_prices = np.concatenate(target_prices)


def targets_pricer() -> np.ndarray:
    # Function used to evaluate the model prices for each target
    all_prices = []
    for ttm, market_slice in surface.slices.items():
        prices = pricer.price_strikes(T=ttm, K=market_slice.strikes, is_calls=market_slice.is_calls)
        all_prices.append(prices)
    return np.concatenate(all_prices)


# Create the calibrator for the model
calibrator = Calibrator(model=model, minimizer=LeastSquares())

# Targets for the calibrator
targets = Targets(target_prices, targets_pricer)
calibrator.add_objective("Targets", targets)

# Calibrate the model
result = calibrator.calibrate()

# Compare the calibrated parameters to the "true" parameters, used to create the synthetic market
calibrated_params = model.get_params()
param_diff = calibrated_params - true_params
print(param_diff)

# Plot the errors from the targets
model.set_params(result.params)

plt.plot(targets.value())
plt.ylabel('Price Error')
plt.show()
