from typing import Dict, Optional, Tuple
import numpy as np

from fypy.calibrate.BaseModelCalibrator import BaseModelCalibrator
from fypy.fit.Minimizer import Minimizer
from fypy.fit.Targets import Targets
from fypy.market.MarketSurface import MarketSurface

from fypy.model.slv.Sabr import Sabr
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.pricing.analytical.SabrHaganOblojPricer import SabrHaganOblojPricer


class SabrModelCalibrator(BaseModelCalibrator):
    def __init__(self,
                 surface: MarketSurface,
                 minimizer: Optional[Minimizer] = None):
        super(SabrModelCalibrator, self).__init__(surface=surface, minimizer=minimizer)

    def calibrate(self,
                  model: Sabr,
                  pricer: Optional[StrikesPricer] = None):

        # Full set of market target prices
        target_vols, weights = self._make_all_targets()

        # Reusable prices vector
        all_prices = np.empty_like(target_vols, dtype=float)

        surface = self.surface
        fwd_curve = self.surface.forward_curve

        def target_voler() -> np.ndarray:
            # Function used to evaluate the model prices for each target
            left = 0
            for ttm, market_slice in surface.slices.items():
                strikes = market_slice.strikes
                num_strikes = len(strikes)
                fwd = fwd_curve.fwd_T(ttm)
                for i in range(num_strikes):
                    all_prices[left + i] = model.implied_vol(K=strikes[i], T=ttm, fwd=fwd)

                left += num_strikes
            return all_prices

        # Create the claibrator for the model
        calibrator = self._init_calibrator(model=model)

        # Targets for the calibrator
        targets = Targets(target_vols, target_voler, weights=weights)
        calibrator.add_objective("Targets", targets)

        # Calibrate the model
        result = self._calibrate(calibrator)

        pricer = pricer or SabrHaganOblojPricer(model=model)
        return result, pricer, None, target_vols

    def _make_all_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        target_vols = []
        weights = []

        for ttm, market_slice in self.surface.slices.items():
            # push back the target vols to fit to
            target_vols.append(market_slice.mid_vols)
            weights.append(np.ones_like(market_slice.mid_vols))

        target_vols = np.concatenate(target_vols)
        weights = np.concatenate(weights)
        return target_vols, weights
