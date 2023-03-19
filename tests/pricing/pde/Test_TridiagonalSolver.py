import unittest

import numpy as np

from fypy.pricing.pde.TridiagonalSolver import solve_dirichlet


# noinspection DuplicatedCode
class Test_TridiagonalSolver(unittest.TestCase):
    def test_homogeneous(self):
        a = np.ones(shape=10)
        b = -2 * np.ones(shape=10)
        c = np.ones(shape=10)
        u = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        s = np.zeros(shape=10)
        for i in range(1, 9):
            s[i] = a[i] * u[i + 1] + b[i] * u[i] + c[i] * u[i - 1]

        u_solved = solve_dirichlet(a=a, b=b, c=c, s=s, u_left=u[0], u_right=u[-1])

        for i in range(10):
            self.assertAlmostEqual(u[i], u_solved[i], delta=0.01)

    def test_inhomogeneous(self):
        a = np.ones(shape=10)
        b = -2 * np.ones(shape=10)
        c = np.ones(shape=10)
        u = np.array([0.0, 0.2, 0.3, 0.35, 0.375, 0.375, 0.35, 0.3, 0.2, 0.0])

        s = np.zeros(shape=10)
        s[0] = b[0] * u[0] + a[0] * u[1]
        for i in range(0, 9):
            s[i] = a[i] * u[i + 1] + b[i] * u[i] + c[i] * u[i - 1]
        s[9] = c[9] * u[8] + b[9] * u[9]

        u_solved = solve_dirichlet(a=a, b=b, c=c, s=s, u_left=u[0], u_right=u[-1])

        for i in range(10):
            self.assertAlmostEqual(u[i], u_solved[i], delta=0.01)

    def test_unsolvable(self):
        a = np.ones(shape=10)
        b = np.ones(shape=10)
        c = np.ones(shape=10)
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2])

        s = np.zeros(shape=10)
        for i in range(1, 9):
            s[i] = a[i] * u[i + 1] + b[i] * u[i] + c[i] * u[i - 1]

        u_solved = solve_dirichlet(a=a, b=b, c=c, s=s, u_left=u[0], u_right=u[-1])

        for i in range(1, 9):
            self.assertTrue(np.isnan(u_solved[i]))
