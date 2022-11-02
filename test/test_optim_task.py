import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.rastrigin import Rastrigin
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchDataItem


class TestOptimizationTask(unittest.TestCase):
    # setUp method is overridden from the parent class SearchDataItem
    def setUp(self):
        self.rastrigin = Rastrigin(3)
        self.optimizationTask = OptimizationTask(self.rastrigin)

        self.perm = np.ndarray(shape=(3), dtype=np.int)
        for i in range(self.perm.size):
            self.perm[i] = self.perm.size-1-i
        self.rastriginPerm = Rastrigin(3)
        self.optimizationTaskPerm = OptimizationTask(self.rastrigin, self.perm)

    def test_InitWithoutPerm(self):
        self.assertEqual(self.optimizationTask.perm.tolist(), [0, 1, 2])
        #self.assertTrue((self.optimizationTask.perm == [0, 1, 2]).all())

    def test_InitWithPerm(self):
        self.assertEqual(self.optimizationTaskPerm.perm.tolist(), [2, 1, 0])

    def test_Calculate(self):
        point = Point([0.0, 0.1, 0.0], [])
        sdi = SearchDataItem(point, -1, 0)
        sdi.functionValues = np.ndarray(shape=(1), dtype=FunctionValue)
        sdi.functionValues[0] = FunctionValue()
        sdi = self.optimizationTask.Calculate(sdi, 0)

        sum: np.double = 0
        for i in range(self.rastrigin.dimension):
            sum += point.floatVariables[i] * point.floatVariables[i] - 10 * math.cos(
                2 * math.pi * point.floatVariables[i]) + 10

        self.assertEqual(sdi.functionValues[0].value, sum)



    #def test_Init(self):
        #self.assertEqual(self.searchDataItem.point.floatVariables, [0.1, 0.5])

    #def test_GetX(self):
        #self.assertEqual(self.searchDataItem.GetX(), 0.3)

# Executing the tests in the above test case class
if __name__ == "__main__":
 unittest.main()