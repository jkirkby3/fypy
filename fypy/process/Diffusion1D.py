from abc import ABC, abstractmethod
from typing import Union
import numpy as np

from fypy.fit.Calibratable import Calibratable


class Diffusion1D(Calibratable, ABC):
    """
    Univariate Diffusion process defined by:
        dS(t) = mu(S,t)*dt + sigma(S,t)*dW(t)
    """

    @abstractmethod
    def mu_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def mu(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self.mu_dt(S=S, t=t, dt=0.)

    @abstractmethod
    def sigma_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def sigma(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self.sigma_dt(S=S, t=t, dt=0.)
