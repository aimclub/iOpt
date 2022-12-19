import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.rastrigin import Rastrigin
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem


class TestOptimizationTask(unittest.TestCase):
    """setUp method is overridden from the parent class OptimizationTask"""

    def setUp(self):
        self.problem = Rastrigin(3)
        self.optimizationTask = OptimizationTask(self.problem)

        self.problemPerm = Rastrigin(3)
        self.perm = np.ndarray(shape=(self.problemPerm.numberOfObjectives + self.problemPerm.numberOfConstraints),
                               dtype=np.int32)
        for i in range(self.perm.size):
            self.perm[i] = self.perm.size - 1 - i
        self.optimizationTaskPerm = OptimizationTask(self.problem, self.perm)

    def test_InitWithoutPermRastrigin3(self):
        self.assertEqual(self.optimizationTask.perm.tolist(), [0])
        # self.assertTrue((self.optimizationTask.perm == [0, 1, 2]).all())

    def test_InitWithPermRastrigin3(self):
        self.assertEqual(self.optimizationTaskPerm.perm.tolist(), [0])

    def test_CalculateRastrigin3(self):
        point = Point([0.5, 0.1, 0.3], [])
        sdi = SearchDataItem(point, -1, 0)
        sdi.functionValues = np.ndarray(shape=(1), dtype=FunctionValue)
        sdi.functionValues[0] = FunctionValue()
        sdi = self.optimizationTask.Calculate(sdi, 0)

        sum: np.double = 0
        for i in range(self.problem.dimension):
            sum += point.floatVariables[i] * point.floatVariables[i] - 10 * math.cos(
                2 * math.pi * point.floatVariables[i]) + 10

        self.assertEqual(sdi.functionValues[0].value, sum)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
