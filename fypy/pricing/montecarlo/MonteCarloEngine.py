from abc import abstractmethod
from typing import List, Optional

import numpy as np

from fypy.pricing.montecarlo.StochasticProcess import StochasticProcess, RandomVariable


class Trajectory:
    def __init__(self, times: np.array, states: np.array):
        self._times = times
        self._states = states

    def get_num_trajectories(self) -> int:
        """
        Get the number of trajectories contained in the Trajectory object.
        """
        return self._states.shape[0]

    def get_times(self) -> np.array:
        """
        Get the time axis of the trajectories.
        """
        return self._times

    def get_component_trajectory(self, i: int, state_index: int) -> np.array:
        """
        Get a single trajectory for a single part of a state.
        """
        return self._states[i, state_index, :]

    def get_state_trajectory(self, i: int) -> np.array:
        """
        Get a single trajectory.
        """
        return self._states[i, :, :]

    def get_states_at_time(self, time: float):
        # Find the first time greater than or equal to the time
        idx = np.searchsorted(self._times, time, side='left')
        return self._states[:, :, idx]


class AdditionalState:
    """
    Class that can observe the state of stochastic variables at every time slice, and evolve additional states, which
    can be recovered at the end of the simulation, and used for pricing.
    """

    @abstractmethod
    def initialize(self, initial_state: np.array, N: int, t0: float):
        """
        Initialize any additional state variables from the visible state.
        """
        raise NotImplementedError

    @abstractmethod
    def evolve(self, last_state: np.array, current_state: np.array, ti: float, tf: float, N: int):
        """
        Given the last state, the current state, and the times, evolve the additional state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """
        Get the additional state.
        """
        raise NotImplementedError


class RunningMaximum(AdditionalState):
    """
    Class that observes the running maximum of a stochastic process.
    """

    def __init__(self):
        self._max = None

    def initialize(self, initial_states: np.array, N: int, t0: float):
        self._max = initial_states

    def evolve(self, last_state: np.array, current_state: np.array, ti: float, tf: float, N: int):
        self._max = np.maximum(self._max, current_state)

    def get_state(self):
        return self._max


class RunningMinimum(AdditionalState):
    """
    Class that observes the running minimum of a stochastic process.
    """

    def __init__(self):
        self._min = None

    def initialize(self, initial_state: np.array, t0: float):
        self._min = initial_state

    def evolve(self, last_state: np.array, current_state: np.array, ti: float, tf: float, N: int):
        self._min = np.minimum(self._min, current_state, out=current_state)

    def get_state(self):
        return self._min


class MonteCarloEngine:
    """
    A simple monte carlo for fast testing.
    """

    def __init__(self,
                 stochastic_process: StochasticProcess,
                 n_paths: int = 1000,
                 dt: float = 1. / 365.25,
                 additional_states: Optional[List[AdditionalState]] = None):
        self._stochastic_process = stochastic_process

        # Any additional states that only depends on the history of the state of the stochastic process at each time,
        # and not on the random variables. This is useful for pricing instruments that depend on the history of the
        # state of the stochastic process, but not on the random variables.
        self._additional_states = additional_states

        self._n_paths = n_paths
        self._dt = dt

        self._save_full_trajectory = False

    def get_additional_states(self) -> Optional[List[AdditionalState]]:
        """
        Get the additional states.
        """
        return self._additional_states

    def add_additional_state(self, additional_state: AdditionalState):
        """
        Add an additional state to the monte carlo engine.
        """
        if self._additional_states is None:
            self._additional_states = []
        self._additional_states.append(additional_state)

    def set_save_full_trajectory(self, save_full_trajectory: bool):
        """
        Set whether to save the full trajectory or just the final state.
        """
        self._save_full_trajectory = save_full_trajectory

    def evolve(self, observation_times: np.array) -> Trajectory:
        """
        Evolve the stochastic process via monte carlo.

        :param observation_times: the times at which to save the state of the stochastic process. Also used to determine
        the range of times over which the process should be evolved. All observation times must be positive, and there
        must be at least one observation time.
        """
        # np.random.seed(self.seed)

        # Validate observation times.
        if len(observation_times) == 0:
            raise RuntimeError("must have at least one observation time")
        observation_times = sorted(observation_times)
        if observation_times[0] < 0:
            raise RuntimeError("observation times must be positive")

        # Make sure time zero is always an observation time.
        if 0 < observation_times[0]:
            observation_times = np.insert(observation_times, 0, 0.0)

        t0 = 0
        t1 = observation_times[-1]

        rv_description = self._stochastic_process.describe_rvs()

        # Number of steps to take
        n_steps = int((t1 - t0) / self._dt) + 1

        state_size = self._stochastic_process.state_size()
        # Initialize the states array.
        if self._save_full_trajectory:
            times = np.zeros(n_steps + 1)
            states_at_t = np.zeros(shape=(self._n_paths, n_steps + 1, state_size))
        else:
            times = np.zeros(len(observation_times))
            states_at_t = np.zeros(shape=(self._n_paths, len(observation_times), state_size))

        # Initialize time variables.
        ti, tf = t0, t0 + self._dt

        # Generate the initial state, save it in the states_at_t array.
        state = self._stochastic_process.generate_initial_state(self._n_paths)
        check_shape = np.shape(state)

        # Make sure the state was the right shape.
        if check_shape != (self._n_paths, state_size):
            raise ValueError(
                f"Initial state has incorrect shape, expected: {(self._n_paths, state_size)}, got: {check_shape}")

        # Initialize any additional states
        if self._additional_states:
            for additional_state in self._additional_states:
                additional_state.initialize(state, self._n_paths, t0)

        # Keep track of what the next observation time is.
        next_obs_time = observation_times[1]
        obs_time_index = 1  # We always save the initial state.
        times[0] = ti  # Save the initial time
        states_at_t[:, 0, :] = state
        for n in range(n_steps):
            # Get or generate new random variables, uses as the random input for state evolution.
            rvs = self._generate_rvs(rv_description)
            new_state = self._stochastic_process.evolve(state, ti, tf, self._n_paths, rvs)

            # If there is an additional state, evolve that too.
            if self._additional_states:
                for additional_state in self._additional_states:
                    additional_state.evolve(state, new_state, ti, tf, self._n_paths)

            state = new_state
            # Save the state.
            if self._save_full_trajectory:
                states_at_t[:, n + 1, :] = state
                times[n + 1] = tf  # Save the next time
            # TODO: Make sure we hit all observation times. Do this by using a smarter time grid.
            elif ti < next_obs_time <= tf:
                # Mark the next observation time as the next observation time after tf.
                while obs_time_index < len(observation_times) and next_obs_time <= tf:
                    # FOR NOW keep copying the same state if there are multiple obs times between ti and tf
                    # This will be fixed by making sure all observation times are part of the time grid.
                    states_at_t[:, obs_time_index, :] = state
                    times[obs_time_index] = next_obs_time
                    obs_time_index += 1
                    if obs_time_index < len(observation_times):
                        next_obs_time = observation_times[obs_time_index]

            # Advance times.
            ti = tf
            tf += self._dt

        # Convert the data to a trajectories object.
        trajectory = np.transpose(states_at_t, (0, 2, 1))
        return Trajectory(times, trajectory)

    def _generate_rvs(self, rv_description: List[RandomVariable]) -> np.array:
        n_rvs = len(rv_description)

        # Right now we only support normal random variables. Pre-generate all the random variables here.
        rvs = np.random.normal(size=(self._n_paths, n_rvs))

        return rvs
