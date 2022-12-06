import math
import unittest

import numpy as np

from iOpt.problems.GKLS import GKLS
from iOpt.trial import FunctionValue, Point


class TestGKLS(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""

    def setUp(self):
        self.GKLS = GKLS(3)

    def test_Calculate(self):
        point = Point([0.9, 0.5, 0.3], [])
        functionValue = FunctionValue()
        functionValue = self.GKLS.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, 0.93113217376043778)

    def test_OptimumValue(self):
        self.assertEqual(self.GKLS.knownOptimum[0].functionValues[0].value, -1.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
