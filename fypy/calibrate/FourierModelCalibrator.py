from typing import Dict, Optional, Tuple
import numpy as np

from fypy.calibrate.BaseModelCalibrator import BaseModelCalibrator
from fypy.calibrate.utils.TargetsWithSmallPriceErr import TargetsWithSmallPriceErr
from fypy.fit.Minimizer import Minimizer
from fypy.market.MarketSlice import MarketSlice
from fypy.market.MarketSurface import MarketSurface
from fypy.model.FourierModel import FourierModel
from fypy.model.levy.LevyModel import LevyModel
from fypy.model.sv.Heston import _HestonBase
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.volatility.implied.ImpliedVolCalculator import black76_vega


class FourierModelCalibrator(BaseModelCalibrator):
    def __init__(self,
                 surface: MarketSurface,
                 minimizer: Optional[Minimizer] = None,
                 do_vega_weight: bool = True):
        super(FourierModelCalibrator, self).__init__(surface=surface, minimizer=minimizer)

        self._do_vega_weight = do_vega_weight

    def calibrate(self,
                  model: FourierModel,
                  pricer: Optional[StrikesPricer] = None):

        if not pricer:
            pricer = ProjEuropeanPricer(model=model,
                                        N=2 ** 12,
                                        L=20 if isinstance(model, _HestonBase) else 15)

        # Full set of market target prices
        target_prices, weights = self._make_all_targets()

        # Reusable prices vector
        all_prices = np.empty_like(target_prices, dtype=float)

        def targets_pricer() -> np.ndarray:
            # Function used to evaluate the model prices for each target
            left = 0
            for ttm, market_slice in self.surface.slices.items():
                num_strikes = len(market_slice.strikes)
                try:
                    pricer.price_strikes_fill(T=ttm, K=market_slice.strikes, is_calls=market_slice.is_calls,
                                              output=all_prices[left:left + num_strikes])
                except Exception as e:
                    print(f'Error getting pricies, filling with nan: {e}')
                    for i in range(num_strikes):
                        all_prices[left + i] = np.nan
                left += num_strikes
            return all_prices

        # Create the claibrator for the model
        calibrator = self._init_calibrator(model=model)

        # Targets for the calibrator
        if isinstance(model, LevyModel):
            targets = TargetsWithSmallPriceErr(target_prices, targets_pricer, weights=weights,
                                               small_price_penalty_mult=10)
        else:
            targets = TargetsWithSmallPriceErr(target_prices, targets_pricer, weights=weights,
                                               small_price_penalty_mult=3)
        calibrator.add_objective("Targets", targets)

        result = self._calibrate(calibrator)

        model_prices = targets_pricer()
        return result, pricer, model_prices, target_prices

    def _make_weights(self, market_slice: MarketSlice) -> np.ndarray:
        if not self._do_vega_weight:
            return np.ones_like(market_slice.mid_prices)

        ttm = market_slice.T
        F = self._fwd_curve(ttm)
        n_strikes = len(market_slice.strikes)

        disc = self._disc_curve(ttm)
        atm_vega = black76_vega(F, F, market_slice.mid_vols[int(n_strikes / 2)], disc=disc, T=ttm)
        return 1. / np.maximum(
            black76_vega(F, market_slice.strikes, market_slice.mid_vols, disc=disc, T=ttm),
            atm_vega / 200)

    def _make_all_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        target_prices = []
        weights = []

        for ttm, market_slice in self.surface.slices.items():
            # push back the target prices to fit to
            target_prices.append(market_slice.mid_prices)

            # Use inverse vega weighting
            weights.append(self._make_weights(market_slice))

        # Full set of market target prices
        target_prices = np.concatenate(target_prices)
        weights = np.concatenate(weights)

        return target_prices, weights
