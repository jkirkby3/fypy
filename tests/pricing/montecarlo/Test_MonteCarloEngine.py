import unittest

import matplotlib.pyplot as plt
import numpy as np

from fypy.pricing.montecarlo.DiffusionStochasticProcess import BlackScholesDiffusion1D, DiffusionStochasticProcess
from fypy.pricing.montecarlo.MonteCarloEngine import MonteCarloEngine, RunningMaximum


class Test_MonteCarlo_Engine(unittest.TestCase):
    def test_price_black_scholes(self):
        r = 0.05
        sigma = 0.2
        S0 = 100.0
        ttm = 1.0
        Z = np.exp(-r * ttm)

        # Create a diffusion process
        diffusion = BlackScholesDiffusion1D(r=r, sigma=sigma)
        # Wrap the diffusion process in a stochastic process, to adapt it to the monte carlo engine
        process = DiffusionStochasticProcess(diffusion=diffusion, S0=S0)

        # Create a monte carlo engine
        engine = MonteCarloEngine(stochastic_process=process, n_paths=100_000, dt=1. / 365.25)
        # Evolve the diffusion process, select what times need to be recorded
        trajectories = engine.evolve(observation_times=np.array([0.25, 0.5, 0.75, ttm]))
        # Get just the final states
        final_states = trajectories.get_states_at_time(1.0)

        # Prices from https://www.mystockoptions.com/black-scholes.cfm

        # Strike 80
        price = np.mean(Z * np.maximum(final_states - 80., 0))
        self.assertAlmostEqual(price, 24.589, delta=0.1)

        # Strike 100
        price = np.mean(Z * np.maximum(final_states - 100., 0))
        self.assertAlmostEqual(price, 10.451, delta=0.1)

        # Strike 120
        price = np.mean(Z * np.maximum(final_states - 120., 0))
        self.assertAlmostEqual(price, 3.247, delta=0.1)

    def test_fixed_strike_lookback_option(self):
        r = 0.05
        sigma = 0.2
        S0 = 100.0
        ttm = 1.0
        Z = np.exp(-r * ttm)

        # Create a diffusion process
        diffusion = BlackScholesDiffusion1D(r=r, sigma=sigma)
        # Wrap the diffusion process in a stochastic process, to adapt it to the monte carlo engine
        process = DiffusionStochasticProcess(diffusion=diffusion, S0=S0)

        # Create a monte carlo engine
        engine = MonteCarloEngine(stochastic_process=process, n_paths=1_000_000, dt=1. / 365.25)

        engine.add_additional_state(RunningMaximum())

        # Evolve the diffusion process, select what times need to be recorded
        trajectories = engine.evolve(observation_times=np.array([ttm]))
        # Just need the additional state
        maxima = engine.get_additional_states()[0].get_state()

        # NOTE: The following is a regression test, I have to separately check the price of the lookback option

        # Strike 80
        price = np.mean(Z * np.maximum(maxima - 80., 0))
        self.assertAlmostEqual(price, 37.50920, delta=0.1)

        # Strike 100
        price = np.mean(Z * np.maximum(maxima - 100., 0))
        self.assertAlmostEqual(price, 18.48461, delta=0.1)

        # Strike 120
        price = np.mean(Z * np.maximum(maxima - 120., 0))
        self.assertAlmostEqual(price, 5.76205, delta=0.1)

    def test_floating_strike_lookback_option(self):
        r = 0.05
        sigma = 0.2
        S0 = 100.0
        ttm = 1.0
        Z = np.exp(-r * ttm)

        # Create a diffusion process
        diffusion = BlackScholesDiffusion1D(r=r, sigma=sigma)
        # Wrap the diffusion process in a stochastic process, to adapt it to the monte carlo engine
        process = DiffusionStochasticProcess(diffusion=diffusion, S0=S0)

        # Create a monte carlo engine
        engine = MonteCarloEngine(stochastic_process=process, n_paths=1_000_000, dt=1. / 365.25)

        engine.add_additional_state(RunningMaximum())

        # Evolve the diffusion process, select what times need to be recorded
        trajectories = engine.evolve(observation_times=np.array([ttm])).get_states_at_time(ttm)
        # Just need the additional state
        maxima = engine.get_additional_states()[0].get_state()

        # NOTE: The following is a regression test, I have to separately check the price of the lookback option

        # Strike 80
        price = np.mean(Z * np.maximum(maxima - trajectories - 10, 0))
        self.assertAlmostEqual(price, 6.17627, delta=0.1)

        # Strike 100
        price = np.mean(Z * np.maximum(maxima - trajectories - 0, 0))
        self.assertAlmostEqual(price, 13.61413, delta=0.1)

        # Strike 120
        price = np.mean(Z * np.maximum(maxima - trajectories + 10, 0))
        self.assertAlmostEqual(price, 23.12642, delta=0.1)


if __name__ == '__main__':
    unittest.main()
