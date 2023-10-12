from typing import List, Union

import numpy as np

from fypy.pricing.montecarlo.StochasticProcess import StochasticProcess, RandomVariable
from fypy.process.Diffusion1D import Diffusion1D


class DiffusionStochasticProcess(StochasticProcess):
    """
    Stochastic process defined by a Diffusion1D process.

    NOTE(Nate): Not sure if this is the best way to do this. It makes more sense to be to have a StochasticProcess *be*
    a diffusion process, but they currently are not. At least for now, while I test this, I leave this as basically
    a wrapper around a diffusion 1D process.
    """

    def __init__(self, S0: float, diffusion: Diffusion1D):
        super().__init__()

        self._S0 = S0
        self._diffusion = diffusion

    def evolve(self, state: np.array, t0: float, t1: float, N: int, dZ: np.array):
        """
        Evolve the process from x0 to x1 using the supplied random variables.
            dS(t) = mu(S,t)*dt + sigma(S,t)*dW(t)
        See Diffusion1D.

        :param state: np.array, the current state of the process, shape (N, self.state_size())
        :param t0: float, initial time
        :param t1: float, final time
        :param N: int, number random scenarios being passed in
        :param dZ: np.array, random variables, shape (N, len(self.describe_rvs()))
        """
        dt = t1 - t0
        dS = self._diffusion.mu_dt(state, t0, dt) + self._diffusion.sigma_dt(state, t0, dt) * dZ
        return state + dS

    def state_size(self) -> int:
        """
        The state size is 1, since this is a 1D diffusion.
        """
        return 1

    def generate_initial_state(self, N: int) -> np.array:
        """
        The initial state is just S0.
        """
        return np.full(shape=(N, 1), fill_value=self._S0)

    def describe_rvs(self) -> List[RandomVariable]:
        """
        A Diffusion1D is driven by a single normal random variable.
        """
        return [RandomVariable.NORMAL]


class BlackScholesDiffusion1D(Diffusion1D):
    """
    A Diffusion1D that recreates the Black-Scholes diffusion model of lognormal diffusion.
    """

    def __init__(self, r: float, sigma: float):
        self._r = r
        self._sigma = sigma

    def mu_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        return self._r * S * dt

    def sigma_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        return self._sigma * S * np.sqrt(dt)

    def get_params(self) -> np.ndarray:
        return np.array([self._r, self._sigma])

    def set_params(self, params: np.ndarray):
        self._r = params[0]
        self._sigma = params[1]

    def num_params(self) -> int:
        return 2
