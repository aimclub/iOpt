import unittest
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from problems.rastrigin import Rastrigin
import test.problems.pointsForTest.rastrigin_points as Sample


class TestRastrigin(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""

    def setUp(self):
        self.problem = Rastrigin(2)

    def test_CalculateAll(self):
        for i in range(0, 99):
            point = Point([Sample.test_points[i][0], Sample.test_points[i][1]], [])
            functionValue = FunctionValue()
            functionValue = self.problem.Calculate(point, functionValue)
            self.assertAlmostEqual(functionValue.value, Sample.test_points[i][2], 7)

    def test_OptimumValue(self):
        self.assertEqual(self.problem.knownOptimum[0].functionValues[0].value, 0.0)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
