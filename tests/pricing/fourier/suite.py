import unittest
from tests.pricing.fourier.Test_Proj_European import Test_Proj_European
from tests.pricing.fourier.Test_Lewis_European import Test_Lewis_European
from tests.pricing.fourier.Test_GilPeleaz_European import Test_GilPeleaz_European
from tests.pricing.fourier.Test_Carr_Madan_European import Test_Carr_Madan_European
from tests.pricing.fourier.Test_Hilbert_European import Test_Hilbert_European


def test_suite():
    suite = unittest.TestSuite()
    for test in (
        Test_Proj_European,
        Test_Lewis_European,
        Test_GilPeleaz_European,
        Test_Carr_Madan_European,
        Test_Hilbert_European,
    ):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(test_suite())
