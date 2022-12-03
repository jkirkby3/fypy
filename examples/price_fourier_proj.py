"""
This example shows how to price using a Fourier pricing method (PROJ)
"""
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.model.levy import *
from fypy.model.sv.Heston import Heston
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
# Create a Model (choose your favorite)
# ============================
model_name = 'Hes'  # Choose this name, then override params if desired

models = {
    'VG': VarianceGamma(sigma=0.2, theta=-0.1, nu=0.6, forwardCurve=fwd, discountCurve=disc_curve),
    'BG': BilateralGamma(alpha_p=1.18, lambda_p=10.57, alhpa_m=1.44, lambda_m=5.57, forwardCurve=fwd,
                         discountCurve=disc_curve),
    'NIG': NIG(alpha=10, beta=-3, delta=0.4, forwardCurve=fwd, discountCurve=disc_curve),
    'CGMY': CMGY(C=0.05, G=4, M=10, Y=1.3, forwardCurve=fwd, discountCurve=disc_curve),
    'MJD': MertonJD(sigma=0.15, lam=0.3, muj=-0.2, sigj=0.3, forwardCurve=fwd, discountCurve=disc_curve),
    'KDE': KouJD(sigma=0.14, lam=2., p_up=0.3, eta1=20, eta2=15, forwardCurve=fwd, discountCurve=disc_curve),
    'BSM': BlackScholes(sigma=0.2, forwardCurve=fwd),
    'Hes': Heston(v_0=0.04, theta=0.04, kappa=0.1, sigma_v=0.5, rho=-0.5, forwardCurve=fwd, discountCurve=disc_curve)
}

# Create Implied Vol calculator
ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd)

# Create the pricers and attach to each model
model = models.get(model_name)
pricer = ProjEuropeanPricer(model=model, N=2 ** 11, L=16 if model_name == 'Hes' else 12)

# Price a set of strikes
strikes = np.arange(50, 180, 1)
is_calls = np.ones(len(strikes), dtype=bool)
prices = pricer.price_strikes(T=T, K=strikes, is_calls=is_calls)

# Plot Prices
plt.plot(strikes, prices, label=model_name)
plt.legend()
plt.xlabel(r'strike, $K$')
plt.ylabel('price')
plt.show()

# Compute Implied Volatilities
ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd)
disc = fwd.discountCurve(T)
vols = ivc.imply_vols(strikes=strikes, prices=prices, is_calls=is_calls, ttm=T)

# Plot Implied Vols
plt.plot(strikes, vols, label=model_name)
plt.legend()
plt.xlabel(r'strike, $K$')
plt.ylabel('implied vol')
plt.show()
