import unittest
from tests.pricing.pde.Test_TridiagonalSolver import Test_TridiagonalSolver


def test_suite():
    suite = unittest.TestSuite()
    for test in (Test_TridiagonalSolver,):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
