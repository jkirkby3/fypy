from abc import ABC, abstractmethod
from typing import Union
import numpy as np

from fypy.process.Diffusion1D import Diffusion1D
from fypy.process.Drift import Drift


class LocalVolProcess1D(Diffusion1D, ABC):
    """
    Univariate Local Vol Diffusion process defined by:
        dS(t) = mu_LV(t)*S(t)*dt + sigma_LV(S,t)*S(t)*dW(t)
              = mu(S,t)*dt + sigma(S,t)*dW(t),

        where   mu(S,t) := mu_LV(t)*S(t)
              sigma(S,t):= sigma_LV(S,t)*S(t)

    Supports log coordinate representation:
        Y(t) = log(S(t))
        dY(t) = (mu_LV(t) - 0.5*sigma_LV(S,t)^2)*dt + sigma_LV(S,t)*dW(t)
    """
    def __init__(self, drift: Drift):
        self._drift = drift

    def mu_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        return S * self._drift.avg(t=t, dt=dt)

    def mu(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return S * self._drift(t)

    def sigma(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return S * self.sigma_LV(S, t)

    def sigma_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        return S * self.sigma_LV_dt(S, t, dt=dt)

    @abstractmethod
    def sigma_LV_dt(self, S: Union[float, np.ndarray], t: float, dt: float) -> Union[float, np.ndarray]:
        """
        Local volatility component:  sigma(S,t):= sigma_LV(S,t)*S(t), averaged over [t, t+dt)
        :param S: float or array, underlying level
        :param t: float, time of evaluation
        :param dt: float, the time step size, we average the vol over [t, t+dt)
        :return: float or array, matches input S
        """
        raise NotImplementedError

    def sigma_LV(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        """
        Local volatility component:  sigma(S,t):= sigma_LV(S,t)*S(t)
        :param S: float or array, underlying level
        :param t: float, time of evaluation
        :return: float or array, matches input S
        """
        return self.sigma_LV_dt(S=S, t=t, dt=0.)

    def log_mu(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self._drift(t) - 0.5 * np.power(self.sigma_LV(S, t), 2)

    def log_mu_dt(self, S: Union[float, np.ndarray], t: float, dt: float) -> Union[float, np.ndarray]:
        return self._drift.avg(t=t, dt=dt) - 0.5 * np.power(self.sigma_LV_dt(S, t=t, dt=dt), 2)

    def log_sigma(self, S: Union[float, np.ndarray], t: float) -> Union[float, np.ndarray]:
        return self.sigma_LV(S, t)

    def log_sigma_dt(self, S: Union[float, np.ndarray], t: float, dt: float) -> Union[float, np.ndarray]:
        return self.sigma_LV_dt(S, t, dt=dt)