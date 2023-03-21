from typing import Callable, Tuple, Dict, Set

import numpy as np
import scipy.integrate as integrate

from fypy.pricing.pde.utility.TridiagonalMatrix import TridiagonalMatrix
from fypy.pricing.pde.utility.TridiagonalSolver import solve_dirichlet


class AsianOption:
    def __init__(self,
                 strike: float,
                 is_call: bool,
                 observation_times: Dict[str, np.array],
                 future_expiries: Dict[str, float],
                 weights: Dict[str, float]):

        self._strike = strike
        self._is_call = is_call

        # Maps from the underlying tag to data. Validate keys.
        all_keys = set(observation_times.keys()).union(future_expiries.keys()).union(weights.keys())
        if len(all_keys) == 0:
            raise ValueError(f"there must be at least one underlying")
        self._num_underlyings = len(all_keys)
        self._all_underlyings = all_keys
        if len(observation_times) != len(all_keys) or len(future_expiries) != len(all_keys) or len(weights) != len(
                all_keys):
            raise ValueError(f"observation_times, future_expiries, and weights must have the same underlying keys")

        self._observation_times = observation_times
        self._future_expiries = future_expiries
        self._weights = weights

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def is_call(self) -> bool:
        return self._is_call

    @property
    def observation_times(self) -> Dict[str, np.array]:
        return self._observation_times

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights

    @property
    def future_expiries(self) -> Dict[str, float]:
        return self._future_expiries

    @property
    def all_underlyings(self) -> Set[str]:
        return self._all_underlyings

    def flatten_times(self):
        underlyings_at_times = {}

        for undl, times in self._observation_times:
            for t in times:
                underlyings_at_times.setdefault(t, set({})).add(undl)
        return underlyings_at_times


class AsianPDEPricer:
    def __init__(self,
                 instrument: AsianOption,
                 q: Callable[[float], float],
                 mu: Callable[[float], float],
                 sigma: Callable[[float, float], float],
                 Ny: int):
        """
        """

        self._instrument = instrument

        self._q = q
        self._mu = mu
        self._sigma = sigma

        self._Ny = Ny

        # Map from times t to \lambda_t. Make sure t = 0 is included.
        self._lambda = {}

        # \Lambda_t = \sum_{t_i} 1_{t_i < t} \lambda_{t_i}
        self._Lambda = {}

        self._create_lambdas()

        self._current_Lambda = self._lambda_grid[-1]

        self._grid_y = None

        # TODO: Set.
        self._y0 = -2.0

        self._A = TridiagonalMatrix.create_matrix(self._Ny)
        self._B_a = np.zeros(shape=self._Ny - 1)
        self._B_b = np.zeros(shape=self._Ny)
        self._B_c = np.zeros(shape=self._Ny - 1)

    def _create_lambdas(self):
        """
        From the observation times and weights, and the drift, mu, we can derive the lambda constants, and the
        cumulative Lambda constants.

        \\lambda_{j,k} = \\theta_j * \\exp [ \\int_{t_{j,k}}^{T_j} \\mu(s) ds ]

        where t_{j,k} is the k-th monitoring type of the j-th maturity, T_j is the final expiry of the j-th maturity.
        """

        self._lambda = {}
        for undl in self._instrument.all_underlyings:
            T = self._instrument.future_expiries[undl]
            w = self._instrument.weights[undl]
            for t in self._instrument.observation_times[undl]:
                # Integrate
                value, err = integrate.quad(self._mu, t, T)
                lm = w * np.exp(value)
                self._lambda.setdefault(T, []).append(lm)

        for t, lambdas in sorted(self._lambda.items()):
            self._lambda[t] = np.sum(lambdas)

        # Create cumulative lambda => Lambda.
        running_sum = 0.
        lambda_array = []
        for t_i, lambda_i in sorted(self._lambda.items()):
            if t_i < 0:
                continue
            running_sum += lambda_i
            lambda_array.append(lambda_i)
            self._Lambda[t_i] = running_sum

        self._lambda_grid = np.array(lambda_array)

    def price(self):
        observation_time_grid = self._create_observation_time_grid()

        # Initialize y-grid.
        self._grid_y = self._create_y_grid()

        # Initialize payout.
        yvals = np.zeros(shape=self._Ny)
        self._initialize_payout(yvals)

        # Evolve PDE.
        for i in range(len(observation_time_grid) - 1):
            # Note that T0 < T1
            T0 = observation_time_grid[i]
            T1 = observation_time_grid[i + 1]

            # Set the current Lambda
            self._current_Lambda = self._Lambda[T1]

            # Create "minor" time grid, for evolution between t0 and t1.
            time_grid = self._create_time_grid(T0, T1)
            for it in range(len(time_grid) - 1):
                t0 = time_grid[it]
                t1 = time_grid[it + 1]
                dt = t0 - t1

                # Set up the A, B matrices.
                self._create_matrices(t0, dt)

                # Update the solution.
                yvals = self._solve_timestep(yvals=yvals)

            # Create next y-grid, interpolate values to initialize the next solution on that grid.
            pass

    def _w(self, t: float, y: float) -> float:
        return -self._mu(t) * (y + self._current_Lambda)

    def _v(self, t: float, y: float) -> float:
        return -self._sigma(t, -self._instrument.strike / y) * np.square(y + self._current_Lambda)

    def _create_y_grid(self) -> np.array:
        return np.linspace(self._y0, self._current_Lambda, self._Ny)

    def _initialize_payout(self, yvals: np.array):
        # NOTE: This probably won't work.
        # for i in range(self._Ny):
        #     # Call payout.
        #     yvals[i] = np.maximum(self._grid_y[i] - self._instrument.strike, 0)
        yvals = np.zeros(shape=self._Ny)

    def _create_matrices(self, t: float, dt: float):
        dy = self._grid_y[1] - self._grid_y[0]  # Uniform grid.
        inv_dy = 1. / dy
        inv_dy_sqr = np.square(inv_dy)
        inv_dt = 1. / dt

        for i, y in enumerate(self._grid_y[1:-1]):  # for j = 1, ..., Ny - 2
            self._A.lower[i + 1] = 0.25 * self._w(t, y) * inv_dy - 0.5 * self._v(t, y) * inv_dy_sqr
            self._A.diag[i + 1] = 0.5 * self._q(t) + inv_dt + 0.5 * self._v(t, y) * inv_dy_sqr
            self._A.lower[i + 1] = -0.25 * self._w(t, y) * inv_dy - 0.5 * self._v(t, y) * inv_dy_sqr

            self._B_a[i + 1] = -self._A.lower[i + 1]
            self._B_b[i + 1] = -self._A.diag[i + 1] + 2. * inv_dt
            self._B_c[i + 1] = -self._A.lower[i + 1]

    def _create_observation_time_grid(self) -> np.array:
        """
        Create a grid of the observation times, ordered from final time to initial time (reverse time ordered).
        """
        times = [0.]
        for t, _ in sorted(self._lambda.items(), reverse=True):
            if 0 < t:
                times.append(t)
        return np.array(times)

    def _create_time_grid(self, t0: float, t1: float) -> np.array:
        """
        Create a time grid for evolving the system between t0 and t1. Note that t1 < t0, since this is a backwards
        equation.
        """

        # TODO: Do this in a smart way.
        dt = 1. / 365.
        return np.arange(t0, t1, dt)

    def _solve_timestep(self, yvals) -> np.array:
        Ay = self._A * yvals

        u_left, u_right = 0., 0.
        return solve_dirichlet(self._B_a, self._B_b, self._B_c, Ay, u_left=u_left, u_right=u_right)
