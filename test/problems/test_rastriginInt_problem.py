import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.rastriginInt import RastriginInt
import test.problems.pointsForTest.rastriginInt_points as Sample

class TestRastriginInt(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""

    def setUp(self):
        self.problem = RastriginInt(3, 2)

    def test_CalculateAll(self):
        for i in range(0, 80):
            point = Point([np.double(Sample.test_points[i][0])],[Sample.test_points[i][1], Sample.test_points[i][2]])

            functionValue = FunctionValue()
            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, np.double(Sample.test_points[i][3]), 6)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
