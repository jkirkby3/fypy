import unittest
from tests.volatility.Test_ImpliedVol_Black76 import Test_ImpliedVol_Black76


def test_suite():
    suite = unittest.TestSuite()
    for test in (Test_ImpliedVol_Black76, ):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
