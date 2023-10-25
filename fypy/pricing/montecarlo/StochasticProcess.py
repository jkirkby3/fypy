from enum import Enum
from typing import List

import numpy as np


class RandomVariable(Enum):
    """ Define the types of random variables that can be requested. """
    NORMAL = 0


class StochasticProcess:
    def __init__(self):
        pass

    def evolve(self, state: np.ndarray, t0: float, t1: float, N: int, dZ: np.ndarray):
        """
        Evolve the process from x0 to x1 using the supplied random variables.
        :param state: np.array, the current state of the process, shape (N, self.state_size())
        :param t0: float, initial time
        :param t1: float, final time
        :param N: int, number random scenarios being passed in
        :param dZ: np.array, random variables, shape (N, len(self.describe_rvs()))
        """
        raise NotImplementedError

    def state_size(self) -> int:
        """
        Return the size of the state vector that the stochstic process evolves.
        """
        raise NotImplementedError

    def generate_initial_state(self, N: int) -> np.array:
        """
        Generate N initial states. If the initial state is deterministic, this can be a single state repeated.
        If the state itself is random, this should be a random sample. For example, if one of the state variables is
        a stochastic volatility, this could be a random sample of the stochastic volatility from some chosen initial
        distribution.
        """
        raise NotImplementedError

    def describe_rvs(self) -> List[RandomVariable]:
        """
        Describe the random variables used for simulation, these are what will be passed to evolve.
        """
        raise NotImplementedError