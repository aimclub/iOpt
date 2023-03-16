import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.stronginC5 import StronginC5
from iOpt.trial import FunctionType


class TestStronginC3(unittest.TestCase):
    """setUp method is overridden from the parent class StronginC5"""

    def setUp(self):
        self.stronginC5 = StronginC5()

    def test_OptimumValue(self):
        self.assertEqual(self.stronginC5.knownOptimum[0].functionValues[0].value, -43.298677234312336)

    def test_CalculateObjective(self):
        point = Point([1.0, 1.0, 1.0, 1.0, 1.0], [])
        res: np.double = 0
        x = point.floatVariables

        res = np.double(math.sin(x[0] * x[2]) - (x[1] * x[4] + x[2] * x[3])*math.cos(x[0] * x[1]))

        functionValue = FunctionValue(FunctionType.OBJECTIV, 5)
        functionValue = self.stronginC5.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_CalculateConstraint1(self):
        point = Point([1.0, 1.0, 1.0, 1.0, 1.0], [])
        res: np.double = 0
        x = point.floatVariables

        res = np.double(-(x[0] + x[1] + x[2] + x[3] + x[4]))

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 0)
        functionValue = self.stronginC5.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_CalculateConstraint2(self):
        point = Point([1.0, 1.0, 1.0, 1.0, 1.0], [])
        res: np.double = 0
        x = point.floatVariables

        res = np.double(x[1] * x[1] / 9 + x[3] * x[3] / 100 - 1.4)

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 1)
        functionValue = self.stronginC5.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_CalculateConstraint3(self):
        point = Point([1.0, 1.0, 1.0, 1.0, 1.0], [])
        res: np.double = 0
        x = point.floatVariables

        res = np.double(3 - pow(x[0] + 1, 2) - pow(x[1] + 2, 2) - pow(x[2] - 2, 2) - pow(x[4] + 5, 2))

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 2)
        functionValue = self.stronginC5.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)
    def test_CalculateConstraint4(self):
        point = Point([1.0, 1.0, 1.0, 1.0, 1.0], [])
        res: np.double = 0
        x = point.floatVariables

        res = np.double(4 * x[0] * x[0] * math.sin(x[0]) + x[1] * x[1] * math.cos(x[1] + x[3]) +
                            x[2] * x[2] * (math.sin(x[2] + x[4]) + math.sin(10 * (x[2] - x[3]) / 3)) - 4)

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 3)
        functionValue = self.stronginC5.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)
    def test_CalculateConstraint5(self):
        point = Point([1.0, 1.0, 1.0, 1.0, 1.0], [])
        res: np.double = 0
        x = point.floatVariables

        res = np.double(x[0] * x[0] + x[1] * x[1] * pow(math.sin((x[0] + x[3]) / 3 + 6.6) +
                                                            math.sin((x[1] + x[4]) / 2 + 0.9), 2)
                            - 17 * pow(math.cos(x[0] + x[2] + 1), 2) + 16)

        functionValue = FunctionValue(FunctionType.CONSTRAINT, 4)
        functionValue = self.stronginC5.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()

