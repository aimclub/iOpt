import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Point
from problems.mco_test1 import mco_test1

from test.problems.pointsForTest import mco_test1_points


class TestMco(unittest.TestCase):
    """setUp method is overridden from the parent class TestHill"""

    def test_Mco1(self):
        problem = mco_test1()
        sample = mco_test1_points
        self.Calculate(problem, sample)

    def Calculate(self, problem, Sample):
        for i in range(0, len(Sample.test_points)):
            fv_point = []
            for j in range(problem.number_of_objectives, problem.number_of_objectives+problem.dimension):
                fv_point.append(np.double(Sample.test_points[i][j]))

            point = Point(fv_point, [])
            for j in range(0, problem.number_of_objectives):
                function_value = FunctionValue()
                function_value.type = FunctionType.OBJECTIV
                function_value.functionID = j

                function_value = problem.calculate(point, function_value)

                self.assertAlmostEqual(function_value.value,
                                       np.double(Sample.test_points[i][j]), 5)

        print(problem.name, "is OK")




"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
