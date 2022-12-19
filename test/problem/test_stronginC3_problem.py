import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.stronginC3 import StronginC3
from iOpt.trial import FunctionType


class TestStronginC3(unittest.TestCase):
    """setUp method is overridden from the parent class StronginC3"""

    def setUp(self):
        self.stronginC3 = StronginC3()

    def test_OptimumValue(self):
        self.assertEqual(self.stronginC3.knownOptimum[0].functionValues[0].value, -1.489444)

    def test_CalculateObjective(self):
        point = Point([1.0, 0.5], [])
        res: np.double = 0
        x1: np.double = point.floatVariables[0]
        x2: np.double = point.floatVariables[1]

        t1: np.double = pow(0.5 * x1 - 0.5, 4.0)
        t2: np.double = pow(x2 - 1.0, 4.0)
        res = np.double(1.5 * x1 * x1 * math.exp(1.0 - x1 * x1 - 20.25 * (x1 - x2) * (x1 - x2)))
        res = np.double(res + t1 * t2 * math.exp(2.0 - t1 - t2))
        res = np.double(-res)

        functionValue = FunctionValue(FunctionType.OBJECTIV, 3)
        functionValue = self.stronginC3.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_CalculateConstraint1(self):
        point = Point([1.0, 0.5], [])
        res: np.double = 0
        x1: np.double = point.floatVariables[0]
        x2: np.double = point.floatVariables[1]

        res = np.double(0.01 * ((x1 - 2.2) * (x1 - 2.2) + (x2 - 1.2) * (x2 - 1.2) - 2.25))

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 0)
        functionValue = self.stronginC3.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_CalculateConstraint2(self):
        point = Point([1.0, 0.5], [])
        res: np.double = 0
        x1: np.double = point.floatVariables[0]
        x2: np.double = point.floatVariables[1]

        res = np.double(100.0 * (1.0 - ((x1 - 2.0) / 1.2) * ((x1 - 2.0) / 1.2) - (x2 / 2.0) * (x2 / 2.0)))

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 1)
        functionValue = self.stronginC3.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_CalculateConstraint3(self):
        point = Point([1.0, 0.5], [])
        res: np.double = 0
        x1: np.double = point.floatVariables[0]
        x2: np.double = point.floatVariables[1]

        res = np.double(10.0 * (x2 - 1.5 - 1.5 * math.sin(6.283 * (x1 - 1.75))))

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 2)
        functionValue = self.stronginC3.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
