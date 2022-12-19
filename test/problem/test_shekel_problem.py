import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.shekel import Shekel


class TestShekel(unittest.TestCase):
    """setUp method is overridden from the parent class Shekel"""

    def setUp(self):
        self.Shekel = Shekel(0)

    def test_Calculate(self):
        point = Point([0.0], [])
        res: np.double = 0
        aShekel = np.array([6.96615, 4.56374, 3.80749, 7.42302, 5.16861, 3.25999, 8.68618, 7.24754, 0.22309, 0.78036],
                           dtype=np.double)
        cShekel = np.array([1.11196, 1.11025, 1.19145, 1.13756, 1.17645, 1.18894, 1.18148, 1.16138, 1.10642, 1.04509],
                           dtype=np.double)
        kShekel = np.array(
            [23.93246, 17.63833, 18.27250, 19.15204, 14.05667, 9.04614, 17.96243, 20.56200, 21.70278, 10.51653],
            dtype=np.double)
        NUM_SHEKEL_COEFF = 10
        res: np.double = 0
        for i in range(NUM_SHEKEL_COEFF):
            res = res - 1 / (kShekel[i] * pow(point.floatVariables[0] - aShekel[i], 2) + cShekel[i])
        functionValue = FunctionValue()
        functionValue = self.Shekel.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_OptimumValue(self):
        self.assertEqual(self.Shekel.knownOptimum[0].functionValues[0].value, -1.85298)

    def test_CalculateOptimumPoint(self):
        point = Point([7.288], [])
        res: np.double = 0
        aShekel = np.array([6.96615, 4.56374, 3.80749, 7.42302, 5.16861, 3.25999, 8.68618, 7.24754, 0.22309, 0.78036],
                           dtype=np.double)
        cShekel = np.array([1.11196, 1.11025, 1.19145, 1.13756, 1.17645, 1.18894, 1.18148, 1.16138, 1.10642, 1.04509],
                           dtype=np.double)
        kShekel = np.array(
            [23.93246, 17.63833, 18.27250, 19.15204, 14.05667, 9.04614, 17.96243, 20.56200, 21.70278, 10.51653],
            dtype=np.double)
        NUM_SHEKEL_COEFF = 10
        res: np.double = 0

        for i in range(NUM_SHEKEL_COEFF):
            res = res - 1 / (kShekel[i] * pow(point.floatVariables[0] - aShekel[i], 2) + cShekel[i])

        functionValue = FunctionValue()
        functionValue = self.Shekel.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
