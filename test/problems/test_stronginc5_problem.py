import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.stronginc5 import Stronginc5
from iOpt.trial import FunctionType
import test.problems.pointsForTest.stronginc5_points as Sample


class TestStronginc5(unittest.TestCase):
    """setUp method is overridden from the parent class Stronginc5"""

    def setUp(self):
        self.problem = Stronginc5()

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([], [])
            for j in range(0, self.problem.dimension):
                point.float_variables = np.append(point.float_variables, Sample.test_points[i][j])
            functionValue = FunctionValue()
            if Sample.test_points[i][5] == 5:
                functionValue.type = FunctionType.OBJECTIV
            else:
                functionValue.type = FunctionType.CONSTRAINT
                functionValue.functionID = Sample.test_points[i][5]

            functionValue = self.problem.calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][6], 6)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()

