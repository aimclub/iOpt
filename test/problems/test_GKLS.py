import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.GKLS import GKLS
import test.problems.pointsForTest.gkls_points as Sample


class TestGKLS(unittest.TestCase):
    """setUp method is overridden from the parent class GKLS"""

    def setUp(self):
        self.problem = GKLS(3, 1)

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([], [])
            for j in range(0, self.problem.dimension):
                point.floatVariables = np.append(point.floatVariables, Sample.test_points[i][j])
            functionValue = FunctionValue()
            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][self.problem.dimension], 5)

    def test_OptimumValue(self):
        self.assertEqual(self.problem.knownOptimum[0].functionValues[0].value, -1.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
