"""
This example shows how to price using a Fourier pricing method (PROJ)
We include two examples:  1) Black Scholes 2) Variance Gamma
"""
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.model.levy.VarianceGamma import *
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76
import matplotlib.pyplot as plt

# ============================
# Set Common Parameters
# ============================
S0 = 100  # Initial stock price
r = 0.01  # Interest rate
q = 0.03  # Dividend yield
T = 1  # Time to maturity of option

# ============================
# Set Term Structures
# ============================
disc_curve = DiscountCurve_ConstRate(rate=r)
div_disc = DiscountCurve_ConstRate(rate=q)
fwd = EquityForward(S0=S0, discount=disc_curve, divDiscount=div_disc)

# ============================
# Create Black-Scholes Model
# ============================
model = BlackScholes(sigma=0.2, forwardCurve=fwd)
pricer = ProjEuropeanPricer(model=model, N=2 ** 10)

# Price a set of strikes
strikes = np.arange(50, 150, 1)
prices = pricer.price_strikes(T=T, K=strikes, is_calls=np.ones(len(strikes), dtype=bool))

# Plot
plt.plot(strikes, prices, label='Black Scholes')

# ============================
# Create Variance Gamma Model
# ============================
model = VarianceGamma(sigma=0.2, theta=0.1, nu=0.8, forwardCurve=fwd, discountCurve=fwd.discountCurve)
pricer = ProjEuropeanPricer(model=model, N=2 ** 10)

# Price a set of strikes
strikes = np.arange(50, 150, 1)
is_calls=np.ones(len(strikes), dtype=bool)
prices = pricer.price_strikes(T=T, K=strikes, is_calls=is_calls)

# Plot Prices
plt.plot(strikes, prices, label='Variance Gamma')
plt.legend()
plt.xlabel(r'strike, $K$')
plt.ylabel('price')
plt.show()

# Compute Implied Volatilities
ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd)
disc = fwd.discountCurve(T)
vols = ivc.imply_vols(strikes=strikes, prices=prices, is_calls=is_calls, ttm=T)

# Plot Implied Vols
plt.plot(strikes, vols, label='Variance Gamma')
plt.legend()
plt.xlabel(r'strike, $K$')
plt.ylabel('implied vol')
plt.show()
