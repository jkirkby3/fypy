from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional


class Calibratable(ABC):
    """
    Generic calibratable interface. Models/model components which may be calibrated must extend this interface
    to be fittable in the framework. This interface simply allows an optimizer to adjust the parameters in a model
    while it fits.
    """
    @abstractmethod
    def set_params(self, params: np.ndarray):
        """
        Sets the parameters in calibratable
        :param params: np.array, the new parameters
        :return: self
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """
        Get the current set of params.
        Note: set params
        :return: np.array, current parameters
        """
        raise NotImplementedError

    @abstractmethod
    def num_params(self) -> int:
        """
        Number of parameters that are settable / calibratable
        :return: int, num params
        """
        raise NotImplementedError

    def param_bounds(self) -> Optional[List[Tuple]]:
        """
        Theoretical parameter bounds. These dont have to be what you used during optimization, but providing reasonable
        bounds here allows a model to be fit out-of-the-box
        :return: list of tuples, upper and lower bounds on each parameter.
        """
        return None

    def default_params(self) -> Optional[np.ndarray]:
        """
        Default set of parameters that can be used as an initial guess, in the absence of better information.
        These dont have to be what you used during optimization (e.g. by inspecting market data first),
        but providing reasonable parameters here allows a model to be fit out-of-the-box
        :return: array of default parameters, by default returns None, to indicate no default parameters provided
        """
        return None
