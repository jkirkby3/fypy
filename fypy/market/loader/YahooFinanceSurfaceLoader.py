from fypy.date.DayCounter import DayCounter, DayCounter_365, Date
from fypy.market.MarketSurface import MarketSlice, MarketSurface

from typing import Tuple, List

import pandas as pd
import numpy as np

from fypy.termstructures.DiscountCurve import EmptyDiscountCurve, DiscountCurve, InterpolatedDiscountCurve
from fypy.termstructures.EquityForward import EquityForward


class YahooFinanceLoader(object):
    def __init__(self,
                 dc: DayCounter = DayCounter_365()
                 ):
        self._dc = dc

    def load_from_file(self,
                       fpath: str,
                       disc_curve: DiscountCurve = EmptyDiscountCurve(),
                       div_disc: DiscountCurve = EmptyDiscountCurve(),
                       fit_discount: bool = False
                       ) -> MarketSurface:
        df = pd.read_csv(fpath)
        return self.load_from_frame(df, disc_curve=disc_curve, div_disc=div_disc, fit_discount=fit_discount)

    def load_from_api(self,
                      ticker: str,
                      disc_curve: DiscountCurve = EmptyDiscountCurve(),
                      div_disc: DiscountCurve = EmptyDiscountCurve(),
                      fit_discount: bool = False
                      ) -> MarketSurface:
        df = self.load_df_from_api(ticker=ticker)
        return self.load_from_frame(df=df, disc_curve=disc_curve, div_disc=div_disc, fit_discount=fit_discount)

    def load_df_from_api(self,
                         ticker: str,
                         volume_filter: int = 0) -> pd.DataFrame:
        import yfinance as yf  # https://github.com/ranaroussi/yfinance
        import requests_cache

        session = requests_cache.CachedSession('yfinance.cache')
        session.headers['User-agent'] = 'my-program/1.0'
        data = yf.Ticker(ticker, session=session)

        # divs = data.dividends

        hist = data.history()['Close']
        spot = hist.iloc[-1]
        date_time = hist.index[-1]
        date = date_time.date()

        expiries = data.options
        all_tenors = data.options

        dfs = []
        for tenor in all_tenors:
            DF_calls, DF_puts = data.option_chain(tenor)
            DF_calls['expiry'] = tenor
            DF_calls['isCall'] = True

            DF_puts['expiry'] = tenor
            DF_puts['isCall'] = False

            dfs.append(DF_calls)
            dfs.append(DF_puts)

        df = pd.concat(dfs)
        df['spot'] = spot
        df['date'] = date

        # Filter the data
        df = df[df['volume'] >= volume_filter]
        df.dropna(how='any', subset=['bid', 'ask'], inplace=True)

        df['ticker'] = ticker
        return df

    def load_from_frame(self,
                        df: pd.DataFrame,
                        disc_curve: DiscountCurve = EmptyDiscountCurve(),
                        div_disc: DiscountCurve = EmptyDiscountCurve(),
                        fit_discount: bool = False
                        ) -> MarketSurface:
        d = df.iloc[0].date
        date = Date.from_str(d) if isinstance(d, str) else Date.from_datetime_date(d)
        spot = df.iloc[0]['spot']
        fwd_curve = EquityForward(S0=spot, discount=disc_curve, divDiscount=div_disc)

        all_tenors = df['expiry'].unique()
        has_discount = False  # 'discount' in df
        has_forward = False  # 'forward' in df

        surface = MarketSurface(forward_curve=fwd_curve, discount_curve=disc_curve)

        ttms = [0]
        discs = [1.0]

        for tenor in all_tenors:
            expiry = Date.from_str(tenor)
            ttm = self._dc.year_fraction(start=date, end=expiry)

            ttms.append(ttm)

            df_tenor = df[df['expiry'] == tenor]
            df_tenor.sort_values('strike', inplace=True)

            strikes = df_tenor['strike'].values
            is_calls = np.asarray(df_tenor['isCall'], dtype=int)

            fwd = df_tenor['forward'] if has_forward else fwd_curve(ttm)
            mids = (df_tenor['bid'].values + df_tenor['ask'].values) / 2

            if fit_discount:
                disc = self._average_discount(spot=spot, ttm=ttm,
                                              strikes=strikes, is_calls=is_calls, prices=mids)
            else:
                disc = df_tenor['discount'] if has_discount else div_disc(ttm)

            discs.append(disc)

            market_slice = MarketSlice(T=ttm, F=fwd, disc=disc, strikes=strikes,
                                       is_calls=is_calls,
                                       bid_prices=df_tenor['bid'].values,
                                       ask_prices=df_tenor['ask'].values,
                                       mid_prices=mids)

            surface.add_slice(ttm=ttm, market_slice=market_slice)

        if fit_discount or has_discount:
            # Sort by ttm
            zipped_lists = zip(ttms, discs)
            sorted_pairs = sorted(zipped_lists)

            tuples = zip(*sorted_pairs)
            ttms, discs = [list(tup) for tup in tuples]

            disc_curve_interp = InterpolatedDiscountCurve.from_log_linear(ttms=ttms, discounts=discs)

            plt_curve = False
            if plt_curve:
                import matplotlib.pyplot as plt
                plt.plot(ttms, disc_curve_interp(ttms))
                plt.show()

            fwd_curve = EquityForward(S0=spot, discount=disc_curve_interp, divDiscount=div_disc)
            surface.forward_curve = fwd_curve
            surface.discount_curve = disc_curve_interp

        return surface

    @staticmethod
    def _average_discount(spot: float,
                          ttm: float,
                          strikes: np.ndarray,
                          is_calls: np.ndarray,
                          prices: np.ndarray) -> float:
        avg = 0
        count = 0
        index = 1
        atm_approx = 0.4 * spot * 0.1 * np.sqrt(ttm)
        min_price = atm_approx / 4
        while index < len(strikes):
            K = strikes[index]

            if K == strikes[index - 1] and (is_calls[index] * is_calls[index - 1] == 0):
                if is_calls[index]:
                    C = prices[index]
                    P = prices[index - 1]
                else:
                    C = prices[index - 1]
                    P = prices[index]

                index += 2
                if C < min_price or P < min_price:
                    continue

                avg += (spot + P - C) / K
                count += 1
            else:
                index += 1

        return avg / count if count > 0 else 1


if __name__ == '__main__':
    tick = 'spy'
    fp = 'C:/temp/spy_2022-10-14.csv'

    loader = YahooFinanceLoader()

    df1 = loader.load_df_from_api(ticker=tick)
    df1.to_csv(fp, index=False)

    df2 = loader.load_from_file(fpath=fp)

    # surf = loader.load_from_api(ticker=tick)
