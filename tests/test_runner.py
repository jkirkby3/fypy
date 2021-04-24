import unittest

from tests.pricing.fourier.suite import test_suite as fourier_suite

##############################################


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests(fourier_suite())
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(test_suite())
