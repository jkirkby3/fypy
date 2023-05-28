from abc import ABC, abstractmethod
from time import time
from typing import Dict, Optional

from fypy.fit.Calibratable import Calibratable
from fypy.fit.Calibrator import Calibrator
from fypy.fit.Loss import LossL2
from fypy.fit.Minimizer import LeastSquares, Minimizer
from fypy.market.MarketSurface import MarketSurface
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.termstructures.ForwardCurve import ForwardCurve


class BaseModelCalibrator(ABC):
    def __init__(self,
                 surface: MarketSurface,
                 minimizer: Optional[Minimizer] = None):

        # Set the surface
        self._fwd_curve: Optional[ForwardCurve] = None
        self._disc_curve: Optional[DiscountCurve] = None
        self.surface = surface

        # Set the minimizer
        self._minimizer = minimizer or LeastSquares(max_nfev=120, ftol=1e-07, xtol=1e-07, gtol=1e-07, verbose=1)

    @property
    def surface(self) -> MarketSurface:
        return self._surface

    @surface.setter
    def surface(self, new_surface: MarketSurface):
        self._surface = new_surface
        self._fwd_curve = new_surface.forward_curve
        self._disc_curve = new_surface.discount_curve

    @abstractmethod
    def calibrate(self,
                  model: Calibratable,
                  pricer: Optional[StrikesPricer] = None):
        raise NotImplementedError

    def _init_calibrator(self, model: Calibratable) -> Calibrator:
        calibrator = Calibrator(model=model, minimizer=self._minimizer)
        calibrator.set_initial_guess(model.get_params())  # So that the params we used above are set as guess
        return calibrator

    def _calibrate(self, calibrator: Calibrator):
        # Calibrate the model
        print("Starting model calibration")
        st = time()
        result = calibrator.calibrate()
        print(f"Done model calibration, cost: {LossL2().aggregate(result.value)}")
        en = time() - st
        print("Calibration Time: ", en)
        return result
