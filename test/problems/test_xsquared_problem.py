import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.xsquared import XSquared


class TestXSquared(unittest.TestCase):
    """setUp method is overridden from the parent class XSquared"""

    def setUp(self):
        self.xsquared = XSquared(3)

    def test_Calculate(self):
        point = Point([1.0, 0.5, 0.3], [])
        sum: np.double = 0
        for i in range(self.xsquared.dimension):
            sum += point.float_variables[i] * point.float_variables[i]

        functionValue = FunctionValue()
        functionValue = self.xsquared.calculate(point, functionValue)
        self.assertEqual(functionValue.value, sum)

    def test_OptimumValue(self):
        self.assertEqual(self.xsquared.known_optimum[0].function_values[0].value, 0.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
