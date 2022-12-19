import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.grishagin import Grishagin


class TesGrishagin(unittest.TestCase):
    """setUp method is overridden from the parent class Grishagin"""

    def setUp(self):
        self.Grishagin = Grishagin(4)

    def test_Calculate(self):
        point = Point([0.066182, 0.582587], [])
        res: np.double = -11.541191160513788

        functionValue = FunctionValue()
        functionValue = self.Grishagin.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_OptimumValue(self):
        self.assertEqual(self.Grishagin.knownOptimum[0].functionValues[0].value, -11.541191160513788)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
