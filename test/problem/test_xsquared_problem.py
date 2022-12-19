import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.xsquared import XSquared


class TestXSquared(unittest.TestCase):
    """setUp method is overridden from the parent class XSquared"""

    def setUp(self):
        self.xsquared = XSquared(3)

    def test_Calculate(self):
        point = Point([1.0, 0.5, 0.3], [])
        sum: np.double = 0
        for i in range(self.xsquared.dimension):
            sum += point.floatVariables[i] * point.floatVariables[i]

        functionValue = FunctionValue()
        functionValue = self.xsquared.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, sum)

    def test_OptimumValue(self):
        self.assertEqual(self.xsquared.knownOptimum[0].functionValues[0].value, 0.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
