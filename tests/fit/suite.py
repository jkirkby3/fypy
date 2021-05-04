import unittest
from tests.fit.Test_Minimizer import Test_Minimizer


def test_suite():
    suite = unittest.TestSuite()
    for test in (Test_Minimizer,):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
