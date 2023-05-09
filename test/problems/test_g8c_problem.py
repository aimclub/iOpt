import unittest
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Point
from problems.g8c import g8c
import test.problems.pointsForTest.g8c_points as Sample

class Testg8c(unittest.TestCase):
    """setUp method is overridden from the parent class g8c"""

    def setUp(self):
        self.problem = g8c()

    def test_CalculateAll(self):
        for i in range(3, 99):
            point = Point([Sample.test_points[i][0], Sample.test_points[i][1]], [])
            functionValue = FunctionValue()
            if Sample.test_points[i][2] == -1:
                functionValue.type = FunctionType.OBJECTIV
            else:
                functionValue.type = FunctionType.CONSTRAINT
                functionValue.functionID = Sample.test_points[i][2]

            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][3], 6)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
