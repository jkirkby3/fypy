
# FyPy

Vanilla and exotic option pricing library to support quantitative R&D. Focus on pricing interesting/useful models 
and contracts (including and beyond Black-Scholes), as well as calibration of financial models to market data.

This library is under active development, although the currently posted features are relatively stable.
## Currently Supported
### Models

- Black-Scholes
- Jump Diffusions: Merton, Kou (Double Exponential)
- Levy: Variance Gamma, Normal Inverse Gaussian (NIG), CGMY/KoBoL, Bilateral Gamma
- Stochastic Volatility: Heston
- SVJ: Bates, Heston + Double Expo Jumps
- SLV: SABR

### Pricing Methods
- Analytical: closed form pricing when available, e.g. Black Scholes
- Fourier: PROJ, Lewis, Gil-Peleaz
- More in progress (PDE, Monte Carlo, etc) ...

### Model Calibration
- Levy Model Calibration (VG, NIG, CGMY, MJD, Kou, etc)
- Heston Stochastic Volatility Model Calibration
- Stochastic Volatility with Jumps Model Calibration
- SABR Model calibration

## Coming Soon !

- Exotic Option Pricing - Asian, Barrier, American, Parisian, Cliquet, Variance Swaps, etc.
- Models: Stochastic Volatility, Regime Switching, Stochastic Local Vol
- Additional pricing methods, such as Mellin Series, PDE, Monte Carlo, etc.
- Regime Switching Calibration
- Credit model calibration 
- Many of the exotic pricing algorithms will be translated into python from:
https://github.com/jkirkby3/PROJ_Option_Pricing_Matlab


## User installation

pip install git+https://github.com/jkirkby3/fypy.git

## Dependencies


fypy requires:

- Python (>= 3.7)
- NumPy (tested with 1.20.2)
- py_lets_be_rational (implied volatility)

## Source code


You can check the latest sources with the command

    git clone https://github.com/jkirkby3/fypy.git

## Test Suite

You can run the full test suite with this command,

    python -m unittest discover -s test -p '*_test.py'

    

## Example: Price Variance Gamma / Black-Scholes Models with PROJ (Fourier) Method

```python

"""
This example shows how to price using a Fourier pricing method (PROJ)
We include two examples:  1) Black Scholes 2) Variance Gamma
"""
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.levy.BlackScholes import *
from fypy.model.levy.VarianceGamma import *
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward
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
model = BlackScholes(sigma=0.2, forwardCurve=fwd, discountCurve=fwd.discountCurve)
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
is_calls = np.ones(len(strikes), dtype=bool)
prices = pricer.price_strikes(T=T, K=strikes, is_calls=is_calls)

# Plot
plt.plot(strikes, prices, label='Variance Gamma')
plt.legend()
plt.xlabel(r'strike, $K$')
plt.ylabel('price')
plt.show()

# Compute Implied Volatilities
ivc = ImpliedVolCalculator_Black76(disc_curve=disc_curve, fwd_curve=fwd)
vols = ivc.imply_vols(strikes=strikes, prices=prices, is_calls=is_calls, ttm=T)

# Plot Implied Vols
plt.plot(strikes, vols, label='Variance Gamma')
plt.legend()
plt.xlabel(r'strike, $K$')
plt.ylabel('implied vol')
plt.show()
```