"""
About: contains pricing/Greeks formulas for black-scholes and black76
"""
import numpy as np
from scipy.stats import norm
from typing import Union


def black76_price(F: float,
                  K: Union[float, np.ndarray],
                  is_call: bool,
                  vol: Union[float, np.ndarray],
                  disc: float,
                  T: float) -> Union[float, np.ndarray]:
    """
    Price strikes of a common parity (ie only call or put). Use black76_price_strikes to price a mix of calls/puts
    :param F: float, forward price
    :param K: float or array, the Strike(s)
    :param is_call: bool, determines if ALL strikes are call or all are put
    :param vol: float or array, the Volatility(ies) ... if float, all strikes get same vol, else a vol smile
    :param disc: float, the discount factor, e.g. 0.99
    :param T: float, time to maturity of option
    :return: float or np.ndarray, same shape as strikes
    """
    vol_st = vol * np.sqrt(T)
    d_1 = (np.log(F / K) + (0.5 * vol ** 2) * T) / vol_st
    d_2 = d_1 - vol_st

    if is_call:
        return disc * (F * norm.cdf(d_1) - norm.cdf(d_2) * K)

    return disc * (norm.cdf(-d_2) * K - F * norm.cdf(-d_1))


def black76_price_strikes(F: float,
                          K: np.array,
                          is_calls: np.ndarray,
                          vol: Union[float, np.ndarray],
                          disc: float,
                          T: float) -> np.ndarray:
    """
    Price strikes of with possibly a mix of call and puts
    :param F: float, forward price
    :param K: float or array, the Strike(s)
    :param is_calls: array of bools, for each strike its true for call or false for put
    :param vol: float or array, the Volatility(ies)  ... if float, all strikes get same vol, else a vol smile
    :param disc: float, the discount factor, e.g. 0.99
    :param T: float, time to maturity of option
    :return: float or np.ndarray, same shape as strikes
    """
    prices = np.zeros(len(is_calls))
    if isinstance(vol, np.ndarray):
        prices[is_calls] = black76_price(F=F, K=K[is_calls], is_call=True, vol=vol[is_calls], disc=disc, T=T)
        prices[~is_calls] = black76_price(F=F, K=K[~is_calls], is_call=False, vol=vol[~is_calls], disc=disc, T=T)
    else:
        prices[is_calls] = black76_price(F=F, K=K[is_calls], is_call=True, vol=vol, disc=disc, T=T)
        prices[~is_calls] = black76_price(F=F, K=K[~is_calls], is_call=False, vol=vol, disc=disc, T=T)

    return prices


def black76_vega(F: float,
                 K: Union[float, np.ndarray],
                 vol: Union[float, np.ndarray],
                 disc: float,
                 T: float) -> Union[float, np.ndarray]:
    """
    Vega(s) for strike(s)
    :param F: float, forward price
    :param K: float or array, the Strike(s)
    :param vol: float or array, the Volatility(ies) ... if float, all strikes get same vol, else a vol smile
    :param disc: float, the discount factor, e.g. 0.99
    :param T: float, time to maturity of option
    :return: float or np.ndarray, same shape as strikes
    """
    vol_st = vol * np.sqrt(T)
    d_1 = (np.log(F / K) + 0.5 * vol_st ** 2) / vol_st
    return disc * F * norm.pdf(d_1) * np.sqrt(T)


def black76_delta(F: float,
                  K: Union[float, np.ndarray],
                  is_call: bool,
                  vol: Union[float, np.ndarray],
                  T: float,
                  div_disc: Optional[float] = 1.0,
                  is_fwd_delta: bool = False) -> Union[float, np.ndarray]:
    """
    Delta for strikes of a common parity (ie only call or put).
    :param F: float, forward price
    :param K: float or array, the Strike(s)
    :param is_call: bool, determines if ALL strikes are call or all are put
    :param vol: float or array, the Volatility(ies) ... if float, all strikes get same vol, else a vol smile
    :param T: float, time to maturity of option
    :param div_disc: float, the dividend discount factor, e.g. 0.99 = exp(-q*T)
    :param is_fwd_delta: bool, if true its a delta w.r.t. forward, else it's w.r.t. spot
    :return: float or np.ndarray, same shape as strikes
    """
    vol_st = vol * np.sqrt(T)
    phi = 1 if is_call else -1
    d_1 = (np.log(F / K) + 0.5 * vol_st ** 2) / vol_st
    delta = phi * norm.cdf(phi * d_1)
    if is_fwd_delta:
        delta *= div_disc
    return delta


def black76_strike_from_delta(F: float,
                              delta: Union[float, np.ndarray],
                              is_call: bool,
                              vol: Union[float, np.ndarray],
                              T: float,
                              div_disc: Optional[float] = 1.0,
                              is_fwd_delta: bool = False
                              ) -> Union[float, np.ndarray]:
    """
    Strike(s) corresponding to delta(s) of a common parity (ie only call or put).
    :param F: float, forward price
    :param delta: float or array, the deltas(s)
    :param is_call: bool, determines if ALL strikes are call or all are put
    :param vol: float or array, the Volatility(ies) ... if float, all strikes get same vol, else a vol smile
    :param T: float, time to maturity of option
    :param div_disc: float, the dividend discount factor, e.g. 0.99 = exp(-q*T)
    :param is_fwd_delta: bool, if true its a delta w.r.t. forward, else it's w.r.t. spot
    :return: float or np.ndarray, same shape as strikes
    """
    phi = 1 if is_call else -1
    vol_st = vol * np.sqrt(T)
    if is_fwd_delta:
        delta /= div_disc
    return F * np.exp(-(vol_st * phi * norm.ppf(phi * delta) - 0.5 * vol * vol * T))


def black_scholes_price(S: float,
                        K: Union[float, np.ndarray],
                        is_call: bool,
                        vol: Union[float, np.ndarray],
                        disc: float,
                        T: float,
                        div_disc: float = 1.0):
    """
    Price strikes of a common parity (ie only call or put). Use black_scholes_price_strikes to price a mix of calls/puts
    :param S: float, spot price
    :param K: float or array, the Strike(s)
    :param is_call: bool, determines if ALL strikes are call or all are put
    :param vol: float or array, the Volatility(ies) ... if float, all strikes get same vol, else a vol smile
    :param disc: float, the discount factor, e.g. 0.99
    :param T: float, time to maturity of option
    :param div_disc: float, the dividen discount factor
    :return: float or np.ndarray, same shape as strikes
    """
    return black76_price(S * div_disc / disc, K, is_call, vol, disc, T)


def black_scholes_price_strikes(S: float,
                                K: np.array,
                                is_calls: np.ndarray,
                                vol: Union[float, np.ndarray],
                                disc: float,
                                T: float,
                                div_disc: float = 1.0) -> np.ndarray:
    """
    Price strikes of with possibly a mix of call and puts
    :param S: float, spot price
    :param K: float or array, the Strike(s)
    :param is_calls: array of bools, for each strike its true for call or false for put
    :param vol: float or array, the Volatility(ies)  ... if float, all strikes get same vol, else a vol smile
    :param disc: float, the discount factor, e.g. 0.99
    :param T: float, time to maturity of option
    :param div_disc: float, the dividen discount factor
    :return: float or np.ndarray, same shape as strikes
    """
    return black76_price_strikes(S * div_disc / disc, K, is_calls=is_calls, vol=vol, disc=disc, T=T)
