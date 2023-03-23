import unittest
from fypy.fit.Minimizer import LeastSquares, ScipyMinimizer
from scipy.optimize import LinearConstraint
import numpy as np


class Test_Minimizer(unittest.TestCase):
    def test_least_squares_quadratic_func(self):
        def fun(x):
            return x

        minimizer = LeastSquares()
        x0 = np.asarray([1, 2, 3, 4])
        bounds = [(-5, 5) for _ in range(len(x0))]
        result = minimizer.minimize(fun, bounds=bounds, guess=x0)

        self.assertTrue(result.success)
        self.assertTrue(len(result.message) > 0)
        self.assertTrue(np.sqrt(np.sum(result.params ** 2)) < 1e-10)

    def test_basic_minimize(self):
        options = {'xtol': 1e-08, 'gtol': 1e-08, 'barrier_tol': 1e-08, 'maxiter': 1000}
        minimizer = ScipyMinimizer(method='trust-constr', options=options)

        fun = lambda x: x[0] ** 2 + x[1] ** 2

        c = LinearConstraint(np.asarray([1, 1]), lb=-np.inf, ub=5)

        result = minimizer.minimize(function=fun,
                                    bounds=[(-3, 3), (-3, 3)],
                                    guess=np.asarray([1, 2]),
                                    constraints=[c])

        self.assertAlmostEqual(0, result.value, 10)
        self.assertAlmostEqual(0, result.params[0], 8)
        self.assertAlmostEqual(0, result.params[1], 8)


if __name__ == '__main__':
    unittest.main()
