"""
About: Pricing Spread option using Kirk 2D method
"""

import math
import scipy.stats as stats

def spread_option_kirk_method(K : float,
                              S0_1 : float,
                              S0_2 : float,
                              T : float,
                              r: float,
                              rho: float,
                              sigma_1: float,
                              sigma_2: float,
                              q_1: float,
                              q_2: float) -> float:
    """
    Calculate price of 2D spread options
    :param K: float, Option Strike
    :param S0_1: float, Initial Price of asset 1
    :param S0_2: float, Initial Price of asset 2
    :param T: float, Time to maturity
    :param r: float, Interest rate
    :param rho: float, correlation coefficient b/w asset 1 and asset 2
    :param sigma_1: float, Vol of asset 1
    :param sigma_2: float, Vol of asset 2
    :param q_1: float, Dividend on asset 1
    :param q_2: float, Dividend on asset 2
    :return float value of option price
    """
    f_1 = S0_1 * math.exp((r - q_1) * T)
    f_2 = S0_2 * math.exp((r - q_2) * T)
    f2_k = f_2 / (f_2 + K)
    sigma_combined = math.sqrt(sigma_1**2 - 2 * rho * sigma_1 * sigma_2 * f2_k + f2_k**2 * sigma_2**2)
    d1 = (math.log(f_1 / (f_1 + K)) + 0.5 * sigma_combined**2 * T) / (sigma_combined * math.sqrt(T))
    d2 = d1 - sigma_combined * math.sqrt(T)
    option_price = math.exp(-r * T) * (f_1 * stats.norm.cdf(d1) - (f_2 + K) * stats.norm.cdf(d2))
    return option_price
