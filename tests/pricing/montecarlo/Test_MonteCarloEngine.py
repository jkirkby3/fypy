import unittest

import matplotlib.pyplot as plt
import numpy as np

from fypy.pricing.montecarlo.DiffusionStochasticProcess import BlackScholesDiffusion1D, DiffusionStochasticProcess
from fypy.pricing.montecarlo.MonteCarloEngine import MonteCarloEngine


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

        # Price a european option from the Trajectories
        x, y, z, tv = [], [], [], []

        for strike in np.linspace(5, 150, 100):
            final_states = trajectories.get_states_at_time(1.0)
            prices = Z * np.maximum(final_states - strike, 0)
            px = np.mean(prices)

            x.append(strike)
            y.append(px)
            z.append(np.maximum(S0 - strike, 0))
            tv.append(px - np.maximum(S0 - strike, 0))

        # Matplotlib figure with two panels in the x direction

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # First panel (left)
        axes[0].plot(x, y,  label="Monte Carlo price for european options")
        axes[0].plot(x, z,  label="Intrinsic value, max(S - K, 0)")
        axes[0].set_xlabel('Strike')
        axes[0].set_ylabel('Price')
        axes[0].set_title('Black Scholes Monte Carlo price')
        axes[0].legend()

        # Second panel (right)
        axes[1].plot(x, tv, label="Time value")
        axes[1].set_xlabel('Strike')
        axes[1].set_ylabel('Time value')
        axes[1].set_title("Option time value")
        axes[1].legend()

        # Adjust the layout to prevent overlapping titles and labels
        plt.tight_layout()

        # Show or save the figure
        plt.show()
        #
        #
        # plt.figure(figsize=(10, 8))
        # plt.plot(x, y, label="Monte Carlo price for european options")
        # plt.plot(x, z, label="Intrinsic value, max(S - K, 0)")
        # plt.xlabel("Strike")
        # plt.ylabel("Price")
        # plt.title("Black Scholes Monte Carlo")
        # plt.legend()
        # plt.show()
        # plt.close("all")
        #
        # plt.figure(figsize=(10, 8))
        # plt.plot(x, tv, label="Time value")
        # plt.xlabel("Strike")
        # plt.ylabel("Time value")
        # plt.title("Option time value")
        # plt.legend()
        # plt.show()




if __name__ == '__main__':
    unittest.main()
