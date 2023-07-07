import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.romeijn2c import Romeijn2c
from iOpt.trial import FunctionType
import test.problems.pointsForTest.romeijn2c_points as Sample


class TestRomeijn2c(unittest.TestCase):
    """setUp method is overridden from the parent class RomeijnC2"""

    def setUp(self):
        self.problem = Romeijn2c()

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([], [])
            for j in range(0, self.problem.dimension):
                point.float_variables = np.append(point.float_variables, Sample.test_points[i][j])
            functionValue = FunctionValue()
            if Sample.test_points[i][6] == 1:
                functionValue.type = FunctionType.OBJECTIV
            else:
                functionValue.type = FunctionType.CONSTRAINT
                functionValue.functionID = Sample.test_points[i][self.problem.dimension]-2

            functionValue = self.problem.calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][7], 7)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
