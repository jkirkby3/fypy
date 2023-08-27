"""
About: Pricing Spread option using Kirk 2D method
"""

import math
import scipy.stats as stats

def spread_option_kirk_method(K, S0_1, S0_2, T, r, rho, sigma_1, sigma_2, q_1, q_2):
    """
    Params:
    K: Option Strike
    S0_1: Initial Price of asset 1
    S0_2: Initial Price of asset 2
    T: Time to maturity
    r: Interest rate
    rho: correlation coefficient b/w asset 1 and asset 2
    sigma_1: Vol of asset 1
    sigma_2: Vol of asset 2
    q_1: Dividend on asset 1
    q_2: Dividend on asset 2
    """
    f_1 = S0_1 * math.exp((r - q_1) * T)
    f_2 = S0_2 * math.exp((r - q_2) * T)
    f2_k = f_2 / (f_2 + K)
    sigma_combined = math.sqrt(sigma_1**2 - 2 * rho * sigma_1 * sigma_2 * f2_k + f2_k**2 * sigma_2**2)
    d1 = (math.log(f_1 / (f_1 + K)) + 0.5 * sigma_combined**2 * T) / (sigma_combined * math.sqrt(T))
    d2 = d1 - sigma_combined * math.sqrt(T)
    option_price = math.exp(-r * T) * (f_1 * stats.norm.cdf(d1) - (f_2 + K) * stats.norm.cdf(d2))
    return option_price
