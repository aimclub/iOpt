import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.GKLS import GKLS


class TestGKLS(unittest.TestCase):
    """setUp method is overridden from the parent class GKLS"""

    def setUp(self):
        self.GKLS = GKLS(3)

    def test_Calculate(self):
        point = Point([0.9, 0.5, 0.3], [])
        sum: np.double = 0.93113217376043778

        functionValue = FunctionValue()
        functionValue = self.GKLS.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, sum)

    def test_OptimumValue(self):
        self.assertEqual(self.GKLS.knownOptimum[0].functionValues[0].value, -1.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
