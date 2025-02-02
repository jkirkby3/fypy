from typing import Optional, Tuple
import numpy as np

from fypy.calibrate.FourierModelCalibrator import FourierModelCalibrator
from fypy.calibrate.utils.TargetsWithSmallPriceErr import TargetsWithSmallPriceErr
from fypy.market.MarketSlice import MarketSlice
from fypy.market.MarketSurface import MarketSurface
from fypy.model.FourierModel import FourierModel
from fypy.model.levy.LevyModel import LevyModel
from fypy.model.sv.Heston import _HestonBase
from fypy.pricing.fourier.ProjPricer import ProjPricer
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.fit.Minimizer import LeastSquares, Minimizer


class MSLevyModelCalibrator(FourierModelCalibrator):
    def __init__(self,
                 surface: MarketSurface,
                 minimizer: Optional[Minimizer] = None,
                 do_vega_weight: bool = True):
        super(MSLevyModelCalibrator, self).__init__(surface=surface, minimizer=minimizer, do_vega_weight=do_vega_weight)


    def calibrate(self, model: LevyModel, pricer: Optional[ProjPricer] = None)-> Tuple:
        model.set_multi_section(multi_section=True)

        # If no pricer is provided, create a default one
        pricer = pricer or ProjEuropeanPricer(
            model=model, N=2 ** 12, L=20 if isinstance(model, _HestonBase) else 15
        )

        slices_list = list(self.surface.slices.items())
        # Number of initial slices for which a higher minimization-precision is required
        first_slices_index = max(1, int(len(slices_list) * 0.3))

        # First stage: calibrate initial maturities with higher precision
        self._multiple_slice_calibration(model, pricer, slices_list[:first_slices_index], high_precision=True)

        # Second stage: calibrate remaining maturities with lower precision
        self._multiple_slice_calibration(model, pricer, slices_list[first_slices_index:-1], high_precision=False)

        # Last slice is calibrated separately without updating frozen parameters
        maturity, market_slice = slices_list[-1]
        result, pricer = self._single_slice_calibration(model, maturity, market_slice, pricer)
        return result, pricer

    def _multiple_slice_calibration(self, model: LevyModel, pricer: ProjPricer, slices: list, high_precision: bool)->None:
        """ Calibrates a set of market slices, updating frozen parameters accordingly. """

        self._minimizer_precision(high_precision=high_precision)

        for maturity, market_slice in slices:
            result, pricer = self._single_slice_calibration(model, maturity, market_slice, pricer)
            pricer.get_model().update_frozen_params(maturity=maturity, parameters=result.params)

    def _single_slice_calibration(self, model: FourierModel, maturity: float, market_slice: MarketSlice,
                                 pricer: Optional[ProjPricer] = None):
        """
        Performs calibration for a single tenor.
        """
        # Prepare market targets and pricer
        target_prices, weights, targets_pricer = self._prepare_targets_and_pricer(
            maturity=maturity, market_slice=market_slice, pricer=pricer)

        # Configure the calibrator with objectives
        calibrator = self._configure_calibrator(model=model, target_prices=target_prices,
                                                targets_pricer=targets_pricer, weights=weights)

        # Perform the calibration process
        result = self._calibrate(calibrator)

        # Get the model prices after calibration
        #model_prices = targets_pricer()

        return result, pricer #, model_prices, target_prices


    def _minimizer_precision(self, high_precision:bool=False)->None:
        self._minimizer= LeastSquares(max_nfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=1) if high_precision else LeastSquares(max_nfev=120, ftol=1e-07, xtol=1e-07, gtol=1e-07, verbose=1)

    def _prepare_targets_and_pricer(self, maturity: float, market_slice: MarketSlice, pricer: ProjPricer):
        """
        Prepares market targets, weights, and the targets pricer function.
        """
        # Generate the full set of market target prices and weights for the single slice
        target_prices, weights = self._make_all_targets(market_slice=market_slice)

        # Create a reusable vector for calculated prices
        all_prices = np.empty_like(target_prices, dtype=float)

        def targets_pricer() -> np.ndarray:
            """
            Evaluates model prices for each target. Handles exceptions by filling the array with NaN values.
            """
            try:
                pricer.price_strikes_fill(T=maturity, K=market_slice.strikes, is_calls=market_slice.is_calls,
                                          output=all_prices)
            except Exception as e:
                print(f'Error getting prices, filling with NaN: {e}')
                for i in range(len(all_prices)):
                    all_prices[i] = np.nan
            return all_prices

        return target_prices, weights, targets_pricer

    def _configure_calibrator(self, model: FourierModel, target_prices: np.ndarray, targets_pricer: callable,
                              weights: np.ndarray):

        # Initialize the calibrator for the model
        calibrator = self._init_calibrator(model=model)

        # Define the small price penalty multiplier based on the model type
        small_price_penalty_mult = 10 if isinstance(model, LevyModel) else 3

        # Create targets for the calibrator
        targets = TargetsWithSmallPriceErr(target_prices, targets_pricer, weights=weights,
                                           small_price_penalty_mult=small_price_penalty_mult)
        calibrator.add_objective("Targets", targets)

        return calibrator



    def _make_all_targets(self, market_slice: MarketSlice=None) -> Tuple[np.ndarray, np.ndarray]:
        target_prices = []
        weights = []

        # push back the target prices to fit to
        target_prices.append(market_slice.mid_prices)

        # Use inverse vega weighting
        weights.append(self._make_weights(market_slice))

        # Full set of market target prices
        target_prices = np.concatenate(target_prices)
        weights = np.concatenate(weights)

        return target_prices, weights

