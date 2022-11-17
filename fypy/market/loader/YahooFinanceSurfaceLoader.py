from fypy.date.DayCounter import DayCounter, DayCounter_365, Date
from fypy.market.MarketSurface import MarketSlice, MarketSurface

from typing import Tuple, List

import pandas as pd
import numpy as np

from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward


class YahooFinanceLoader(object):
    def __init__(self,
                 dc: DayCounter = DayCounter_365()
                 ):
        self._dc = dc

    def load_from_file(self, fpath: str) -> MarketSurface:
        df = pd.read_csv(fpath)
        return self.load_from_frame(df)

    def load_from_api(self,
                      ticker: str,
                      disc_curve=DiscountCurve_ConstRate(rate=0),
                      div_disc=DiscountCurve_ConstRate(rate=0)
                      ) -> MarketSurface:
        df = self.load_df_from_api(ticker=ticker)
        return self.load_from_frame(df=df, disc_curve=disc_curve, div_disc=div_disc)

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
        spot = hist.iloc[len(hist) - 1]
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

        df['ticker'] = ticker
        return df

    def load_from_frame(self,
                        df: pd.DataFrame,
                        disc_curve=DiscountCurve_ConstRate(rate=0),
                        div_disc=DiscountCurve_ConstRate(rate=0)
                        ) -> MarketSurface:
        d = df.iloc[0].date
        date = Date.from_str(d) if isinstance(d, str) else Date.from_datetime_date(d)
        spot = df.iloc[0]['spot']
        fwd_curve = EquityForward(S0=spot, discount=disc_curve, divDiscount=div_disc)

        all_tenors = df['expiry'].unique()
        has_discount = 'discount' in df
        has_forward = 'forward' in df

        surface = MarketSurface(forward_curve=fwd_curve, discount_curve=disc_curve)

        for tenor in all_tenors:
            expiry = Date.from_str(tenor)
            ttm = self._dc.year_fraction(start=date, end=expiry)

            df_tenor = df[df['expiry'] == tenor]
            strikes = df_tenor['strike'].values
            is_calls = np.asarray(df_tenor['isCall'], dtype=int)

            fwd = df_tenor['forward'] if has_forward else fwd_curve(ttm)
            disc = df_tenor['discount'] if has_discount else div_disc(ttm)

            market_slice = MarketSlice(T=ttm, F=fwd, disc=disc, strikes=strikes,
                                       is_calls=is_calls,
                                       bid_prices=df_tenor['bid'].values,
                                       ask_prices=df_tenor['ask'].values)

            surface.add_slice(ttm=ttm, market_slice=market_slice)

        return surface


if __name__ == '__main__':
    tick = 'spy'
    fp = 'C:/temp/spy_2022-10-14.csv'

    loader = YahooFinanceLoader()

    df1 = loader.load_df_from_api(ticker=tick)
    df1.to_csv(fp, index=False)

    df2 = loader.load_from_file(fpath=fp)

    # surf = loader.load_from_api(ticker=tick)
