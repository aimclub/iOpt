import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.rastrigin import Rastrigin


class TestRastrigin(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""

    def setUp(self):
        self.rastrigin = Rastrigin(3)

    def test_Calculate(self):
        point = Point([1.0, 0.5, 0.3], [])
        sum: np.double = 0
        for i in range(self.rastrigin.dimension):
            sum += point.floatVariables[i] * point.floatVariables[i] - 10 * math.cos(
                2 * math.pi * point.floatVariables[i]) + 10

        functionValue = FunctionValue()
        functionValue = self.rastrigin.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, sum)

    def test_OptimumValue(self):
        self.assertEqual(self.rastrigin.knownOptimum[0].functionValues[0].value, 0.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
