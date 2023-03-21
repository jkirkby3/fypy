import unittest
import numpy as np

from fypy.pricing.pde.utility.TridiagonalMatrix import TridiagonalMatrix


class Test_TridiagonalMatrix(unittest.TestCase):
    def test__identity_case(self):
        upper = np.zeros(shape=4)
        diag = np.ones(shape=5)
        lower = np.zeros(shape=4)

        Id = TridiagonalMatrix(upper=upper, diag=diag, lower=lower)
        self.assertEqual(len(Id), 5)

        vec = np.array([1.1, 2.3, 8.5, 3.6, 4.9])

        out = Id * vec

        for check, expected in zip(out, vec):
            self.assertAlmostEqual(check, expected, delta=1.e-4)

    def test__general_cast(self):
        upper = np.ones(shape=4)
        diag = -2 * np.ones(shape=5)
        lower = np.ones(shape=4)

        M = TridiagonalMatrix(upper=upper, diag=diag, lower=lower)
        self.assertEqual(len(M), 5)

        vec = np.ones(shape=5)

        out = M * vec
        expected = np.array([-1., 0., 0., 0., -1.])
        for check, expected in zip(out, expected):
            self.assertAlmostEqual(check, expected, delta=1.e-4)

        vec = np.array([1.1, 2.3, 8.5, 3.6, 4.9])
        out = M * vec
        expected = np.array([0.1, 5, -11.1, 6.2, -6.2])
        for check, expected in zip(out, expected):
            self.assertAlmostEqual(check, expected, delta=1.e-4)
