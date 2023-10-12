import unittest

import matplotlib.pyplot as plt
import numpy as np

from fypy.pricing.montecarlo.DiffusionStochasticProcess import BlackScholesDiffusion1D, DiffusionStochasticProcess
from fypy.pricing.montecarlo.MonteCarloEngine import MonteCarloEngine


class Test_MonteCarlo_Engine(unittest.TestCase):
    def test_price_black_scholes(self):
        diffusion = BlackScholesDiffusion1D(r=0.05, sigma=0.2)
        process = DiffusionStochasticProcess(diffusion=diffusion, S0=100.0)
        engine = MonteCarloEngine(stochastic_process=process, n_paths=1_000, dt=1. / 365.25)
        trajectories = engine.evolve(observation_times=np.array([0.25, 0.5, 0.75, 1.0]))


        plt.figure(figsize=(10, 8))
        for i in range( trajectories.get_num_trajectories()):
            plt.plot(trajectories.get_times(), trajectories.get_component_trajectory(i, 0))
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Black Scholes Monte Carlo")
        plt.show()


if __name__ == '__main__':
    unittest.main()
