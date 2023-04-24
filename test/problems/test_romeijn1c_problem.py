import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.romeijn1c import Romeijn1c
from iOpt.trial import FunctionType
import test.problems.pointsForTest.romeijn1c_points as Sample


class TestRomeijn1c(unittest.TestCase):
    """setUp method is overridden from the parent class Romeijn1c"""

    def setUp(self):
        self.problem = Romeijn1c()

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([], [])
            for j in range(0, self.problem.dimension):
                point.floatVariables = np.append(point.floatVariables, Sample.test_points[i][j])
            functionValue = FunctionValue()
            if Sample.test_points[i][self.problem.dimension] == 1:
                functionValue.type = FunctionType.OBJECTIV
            else:
                functionValue.type = FunctionType.CONSTRAINT
                functionValue.functionID = Sample.test_points[i][self.problem.dimension] - 2

            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][self.problem.dimension+1], 6)



"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
