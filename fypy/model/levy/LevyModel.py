from abc import ABC, abstractmethod
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from fypy.model.FourierModel import FourierModel
from typing import Union, Optional, Dict
import numpy as np
from contextlib import contextmanager


class LevyModel(FourierModel, ABC):
    def __init__(self,
                 forwardCurve: ForwardCurve,
                 discountCurve: DiscountCurve,
                 params: np.ndarray,):
        """
        Base class for an exponential Levy model, which is a model for which the characteristic function is known, hence
        enabling pricing by Fourier methods. These models are defined uniquely by their Levy "symbol", which determines
        the chf by:   chf(T,xi) = exp(T*symbol(xi)),  for all T>=0

        :param forwardCurve: Forward curve term structure
        """
        super().__init__(discountCurve=discountCurve,
                         forwardCurve=forwardCurve)

        self._params = params
        self._is_multi_section = False
        self.chf = self._chf_levy  # Default to the simple characteristic function


    @abstractmethod
    def symbol(self, xi: Union[float, np.ndarray]):
        """
        Levy symbol, uniquely defines Characteristic Function via: chf(T,xi) = exp(T * symbol(xi)), for all T>=0
        :param xi: np.ndarray or float, points in frequency domain
        :return: np.ndarray or float, symbol evaluated at input points in frequency domain
        """
        raise NotImplementedError



    @abstractmethod
    def convexity_correction(self) -> float:
        """
        Computes the convexity correction for the Levy model, added to log process drift to ensure
        risk neutrality
        """
        raise NotImplementedError


    def chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Placeholder method that is dynamically overridden during initialization.
        Necessary to define the abstract method.
        """
        raise NotImplementedError("chf method is dynamically assigned during initialization.")

    def _chf_levy(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function for single-section Levy models.
        :param T: Time to maturity
        :param xi: Points in the frequency domain
        :return: Characteristic function evaluated at the given points
        """
        return np.exp(T * self.symbol(xi))



    def _chf_multi_section(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Characteristic function for multi-section Levy models

        :param T: Time to maturity
        :param xi: Points in the frequency domain
        :return: Characteristic function evaluated at the given points
        """

        if T >= self._last_tenor:
            return np.exp((T - self._last_tenor) * self.symbol(xi)) * self._compute_frozen_chf(T=self._last_tenor, xi=xi)

        return self._compute_frozen_chf(T=T, xi=xi)

    def _compute_frozen_chf(self, T: float, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes the characteristic function for multi-section Levy models, iterating over the frozen parameters.

        :param T: Time up to which we compute the characteristic function
        :param xi: Points in the frequency domain
        :return: Characteristic function evaluated at the given points
        """
        # TODO : Store CHF values along the grid instead of recomputing them each time
        T_previous = 0
        chf_value = 1.0

        for frozen_maturity, frozen_params in sorted(self._frozen_params.items()):
            # TODO : Double-check self._last_tenor case
            if T < frozen_maturity:
                with self.temporary_params(frozen_params):
                    return np.exp((T - T_previous) * self.symbol(xi)) * chf_value

            with self.temporary_params(frozen_params):
                chf_value *= np.exp((frozen_maturity - T_previous) * self.symbol(xi))

            T_previous = frozen_maturity

        return chf_value



    def set_multi_section(self, multi_section: bool, frozen_params: Optional[Dict[float, list]] = None):
        """
        Configures the model as either multi-section or single-section.

        :param multi_section: If True, configures the model as multi-section
        :param frozen_params: Dictionary of frozen parameters for multi-section models
        """
        self._is_multi_section = multi_section
        if multi_section:
            self._frozen_params = frozen_params if frozen_params else {}
            self._last_tenor= max(self._frozen_params) if frozen_params else 0
            self.chf = self._chf_multi_section
        else:
            self._frozen_params = None # In case the object was previously created as multi-section
            self.chf = self._chf_levy

    def update_frozen_params(self, maturity:float, parameters:list):
        if self._is_multi_section:
            self._frozen_params[maturity]=parameters
            self._last_tenor= max(self._frozen_params)
        else:
            TypeError("Lévy Model is not a multi-section one")


    def risk_neutral_log_drift(self) -> float:
        """ Compute the risk-neutral drift of log process """
        return self.forwardCurve.drift(0, 1) + self.convexity_correction()

    def set_params(self, params: np.ndarray):
        self._params = params
        return self

    def get_params(self) -> np.ndarray:
        return self._params

    def get_frozen_params(self) ->Dict:
        return self._frozen_params


    @contextmanager
    def temporary_params(self, new_params: np.ndarray):
        """
        This function temporarily sets the model's parameters (`_params`) to `new_params` for the duration of the
        context, and then automatically restores the original parameters after exiting the context.
        This method could be used for inhomogeneous Levy models, where the model's parameters must be changed frequently.

        :param new_params: np.ndarray, new set of parameters to use temporarily.
        """
        # Store the original parameters before modifying them
        original_params = self.get_params()
        try:
            # Set the new parameters
            self.set_params(new_params)
            yield
        finally:
            # Restore the original parameters after the context is done
            self.set_params(original_params)



