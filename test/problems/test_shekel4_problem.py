import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.shekel4 import Shekel4
import test.problems.pointsForTest.shekel4_points as Sample


class TestShekel4(unittest.TestCase):
    """setUp method is overridden from the parent class Shekel4"""

    def setUp(self):
        self.fn = 1
        self.problem = Shekel4(self.fn)

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([], [])
            for j in range(0, self.problem.dimension):
                point.floatVariables = np.append(point.floatVariables, Sample.test_points[i][j])
            functionValue = FunctionValue()
            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][self.problem.dimension], 7)

    def test_OptimumValue(self):
        pointfv = np.ndarray(shape=(self.problem.dimension), dtype=np.double)
        pointfv.fill(4)
        point = Point(pointfv, [])
        functionValue = FunctionValue()
        functionValue = self.problem.Calculate(point, functionValue)
        self.assertEqual(self.problem.knownOptimum[0].functionValues[0].value, functionValue.value)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
