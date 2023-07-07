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
                point.float_variables = np.append(point.float_variables, Sample.test_points[i][j])
            functionValue = FunctionValue()
            functionValue = self.problem.calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][self.problem.dimension], 5)

    def test_OptimumValue(self):
        self.assertEqual(self.problem.known_optimum[0].function_values[0].value, -1.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
