import unittest
from fypy.fit.Minimizer import LeastSquares
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


if __name__ == '__main__':
    unittest.main()
