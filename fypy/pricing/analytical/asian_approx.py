import numpy as np
from scipy.stats import norm


def asian_vorst_approx_price(
        S_0: float,
        sigma: float,
        M: int,
        K: float,
        is_call: bool,
        T: float,
        r: float,
        q: float,
        include_spot_in_average: bool = True
) -> float:
    """
    Price an Asian option using Vorst's approximation.  
    
    :param S_0: float, initial stock price (e.g. 100)
    :param sigma: float, volatility of diffusion (e.g. 0.2)
    :param M: int, number of subintervals of [0,T] (total of M+1 monitoring points in time grid, including S_0)
    :param K: float, strike  (e.g. 100)
    :param is_call: bool, true for call (else put)
    :param T: float, time remaining until maturity (in years, e.g. T=1)
    :param r: float, interest rate (e.g. 0.05)
    :param q: float, dividend yield (e.g. 0.05). 
        NOTE: To price an option on a future, use q = r, so that the drift is zero
    :param include_spot_in_average: bool, enforces convention that S_0 is included in the average,
        else averaging starts at S_1
    :return: float, price of the Asian option using Vorst approximation
    """
    dt = T / M

    if include_spot_in_average:
        K = (M + 1) / M * K - S_0 / M  # Adjustmentfor avg convention

    mu_G = np.log(S_0) + (r - q - 0.5 * sigma ** 2) * (T + dt) / 2
    sigma_G = np.sqrt(sigma ** 2 * (dt + (T - dt) * (2 * M - 1) / (6 * M)))

    if r - q == 0:
        mult = M
    else:
        mult = np.exp((r - q) * dt) * (1 - np.exp((r - q) * M * dt)) / (1 - np.exp((r - q) * dt))

    E_A = (S_0 / M) * mult
    E_G = np.exp(mu_G + 0.5 * sigma_G ** 2)
    K_adj = K - (E_A - E_G)  # Adjust strike based on difference between arithmetic and geometric

    d1 = (mu_G - np.log(K_adj) + sigma_G ** 2) / sigma_G
    d2 = d1 - sigma_G

    price = np.exp(-r * T) * (np.exp(mu_G + 0.5 * sigma_G ** 2) * norm.cdf(d1) - K_adj * norm.cdf(d2))

    # % Final adjustment so due to different averagin convention,  Avg(S_0,S_1,...,S_M) instead of Avg(S_1,...,S_M)
    if include_spot_in_average:
        price = price * (M / (M + 1))

    if is_call != 1:  # % Put Option
        if include_spot_in_average:
            if r - q == 0:
                mult = M + 1
            else:
                mult = (np.exp((r - q) * T * (1 + 1 / M)) - 1) / (np.exp((r - q) * dt) - 1)

            price = price - S_0 / (M + 1) * np.exp(-r * T) * mult + K * np.exp(-r * T)  # NOTE: we use ORIGINAL strike

        else:
            price = price - S_0 / M * np.exp(-r * T) * mult + K * np.exp(-r * T)  # NOTE: we use ORIGINAL strike

    return price
