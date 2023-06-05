import unittest

import test.problems.pointsForTest.stronginc3_points as Sample
from iOpt.trial import FunctionType
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.stronginc3 import Stronginc3


class TestStronginc3(unittest.TestCase):
    """setUp method is overridden from the parent class Stronginc3"""

    def setUp(self):
        self.problem = Stronginc3()

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([Sample.test_points[i][0], Sample.test_points[i][1]], [])
            functionValue = FunctionValue()
            if Sample.test_points[i][2] == 3:
                functionValue.type = FunctionType.OBJECTIV
            else:
                functionValue.type = FunctionType.CONSTRAINT
                functionValue.functionID = Sample.test_points[i][2]

            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][3], 6)



"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
