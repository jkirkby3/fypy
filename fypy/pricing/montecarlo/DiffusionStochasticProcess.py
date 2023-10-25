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

    def evolve(self, state: np.ndarray, t0: float, t1: float, N: int, dZ: np.ndarray):
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

    def __init__(self, r: float, sigma: float, q: float = 0):
        self._mu = r - q
        self._q = q
        self._sigma = sigma

    def mu_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        return self._mu * S * dt

    def sigma_dt(self, S: Union[float, np.ndarray], t: float, dt: float = 0.) -> Union[float, np.ndarray]:
        return self._sigma * S * np.sqrt(dt)

    def get_params(self) -> np.ndarray:
        return np.array([self._mu, self._sigma])

    def set_params(self, params: np.ndarray):
        self._mu = params[0] - self._q
        self._sigma = params[1]

    def num_params(self) -> int:
        return 2


class HestonSLV(StochasticProcess):
    """
    Heston model as a stochastic process
    """

    def __init__(self,
                 S0: float,
                 r: float,
                 v_0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 2.,
                 sigma_v: float = 0.3,
                 rho: float = -0.6):
        super().__init__()
        self._S0 = S0

        self._r = r
        self._v0 = v_0
        self._theta = theta
        self._rho = rho
        self._kappa = kappa
        self._xi = sigma_v

    def _mu_dt(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Drift terms for the spot and vol processes
        """
        S = state[:, 0]
        v = state[:, 1]
        return np.array([self._r * S * dt, self._kappa * (self._theta - v) * dt]).transpose()

    def _sigma_dt(self, state: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Diffusion terms for the spot and vol processes
        """
        S = state[:, 0]
        var = state[:, 1]
        vol = np.sqrt(var)
        sqdt = np.sqrt(dt)
        return np.array([vol * S * sqdt, self._xi * vol * sqdt]).transpose()

    def _correlate(self, dZ: np.ndarray) -> np.array:
        """
        Correlate the random variables so the correlation matches the Heston model parameter.
        """
        dW = dZ[:, 0]
        dZ = dZ[:, 1]
        return np.array([dW, self._rho * dW + np.sqrt(1 - self._rho ** 2) * dZ]).transpose()

    def evolve(self, state: np.array, t0: float, t1: float, N: int, dZ: np.array):
        """
        Evolve the process from x0 to x1 using the supplied random variables.
            dS_t = r * S_t * dt + sqrt(v) * S_t * dW(t)
            dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) * dZ
        """
        dt = t1 - t0
        dZ = self._correlate(dZ)
        dS = self._mu_dt(state, t0, dt) + self._sigma_dt(state, t0, dt) * dZ
        return state + dS

    def state_size(self) -> int:
        """
        Two state variables: spot and vol.
        """
        return 2

    def generate_initial_state(self, N: int) -> np.array:
        """
        The initial state is just S0.
        """
        return np.full(shape=(N, 2), fill_value=(self._S0, self._v0))

    def describe_rvs(self) -> List[RandomVariable]:
        """
        Two variables needed: dW1 and dW2
        """
        return [RandomVariable.NORMAL, RandomVariable.NORMAL]
