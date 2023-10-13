import numpy as np

from fypy.date.Date import Date
from fypy.pricing.montecarlo.MonteCarloEngine import Trajectory, MonteCarloEngine


class TrajectoryPricer:
    """
    Class that knows how to price an instrument from a monte carlo trajectory object.
    """

    def price(self, trajectories: Trajectory) -> np.array:
        """
        Price the instrument from the trajectories.
        """
        raise NotImplementedError

    def price_with_engine(self, engine: MonteCarloEngine):
        """
        Price using a monte carlo engine to generate trajectories.
        """
        # TODO: Validate that the stochastic process inside the engine is compatible with this pricer.
        obs_times = self.get_observation_times()
        trajectories = engine.evolve(observation_times=obs_times[-1])
        return self.price(trajectories=trajectories)

    # TODO(Nate): Really, this type of thing needs to go in an instrument definition, or at least we need to *extract*
    #   the observation times from the instrument definition.
    def get_observation_times(self) -> np.array:
        """
        Get the observation times of the instrument. This is used to determine at which points in the trajectory the
        MonteCarloEngine should save the stochastic process's state.
        """
        raise NotImplementedError
