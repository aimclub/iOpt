import unittest

from unittest import mock
from unittest.mock import Mock

from typing import List


from iOpt.method.optim_task import OptimizationTask
from iOpt.method.calculator import Calculator
from iOpt.trial import Point
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.search_data import SearchData
from iOpt.method.mixed_integer_method import MixedIntegerMethod


class TestMixedIntegerMethod(unittest.TestCase):

    @mock.patch('iOpt.problem.Problem')
    @mock.patch('iOpt.evolvent.evolvent')
    def setUp(self, mock_evolvent, mock_problem):
        mock_problem.numberOfFloatVariables = 1
        mock_problem.numberOfDisreteVariables = 2
        mock_problem.discreteVariableValues = [["A", "B"], ["A", "B"]]
        mock_problem.lowerBoundOfFloatVariables = [[-0.2]]
        mock_problem.upperBoundOfFloatVariables = [[2.0]]
        mock_problem.numberOfObjectives = 1
        mock_problem.numberOfConstraints = 0
        task = OptimizationTask(mock_problem)
        self.mixedIntegerMethod = MixedIntegerMethod(SolverParameters(), task,
                                                     mock_evolvent, SearchData(mock_problem))

    def mock_GetImage(self, x):
        return [x]

    def mock_CalculateFunctionals(self, searchDataItem: SearchDataItem):
        searchDataItem.SetZ(searchDataItem.GetX() * 0.34)
        searchDataItem.SetIndex(0)
        return searchDataItem

    def mock_CalculateFunctionalsForItems(self, items: List[SearchDataItem]):
        for item in items:
            self.mock_CalculateFunctionals(item)
        return items

    def test_FirstIteration(self):
        self.mixedIntegerMethod.CalculateFunctionals = Mock(side_effect=self.mock_CalculateFunctionals)
        self.mixedIntegerMethod.evolvent.GetImage = Mock(side_effect=self.mock_GetImage)
        self.mixedIntegerMethod.FirstIteration()
        self.assertEqual(self.mixedIntegerMethod.searchData.GetCount(), 9)
        self.mixedIntegerMethod.CalculateFunctionals.assert_called()
        self.mixedIntegerMethod.evolvent.GetImage.assert_called()

    def test_FirstIterationParallel(self):
        self.mixedIntegerMethod.parameters.numberOfParallelPoints = 5
        self.mixedIntegerMethod.evolvent.GetImage = Mock(side_effect=self.mock_GetImage)
        calculator = Calculator(None, self.mixedIntegerMethod.parameters)
        calculator.CalculateFunctionalsForItems = Mock(side_effect=self.mock_CalculateFunctionalsForItems)
        self.mixedIntegerMethod.FirstIteration(calculator)
        self.assertEqual(self.mixedIntegerMethod.searchData.GetCount(), 13)
        calculator.CalculateFunctionalsForItems.assert_called()
        self.mixedIntegerMethod.evolvent.GetImage.assert_called()

    @mock.patch('iOpt.method.method.Method.CalculateGlobalR')
    def test_CalculateIterationPoint(self, mock_CalculateGlobalR):
        first = SearchDataItem(Point([0.0], ["A", "A"]), 0.0, discreteValueIndex=0)
        last = SearchDataItem(Point([1.0], ["B", "B"]), 4.0, discreteValueIndex=3)
        last.globalR = 2.3
        self.mixedIntegerMethod.searchData.InsertFirstDataItem(first, last)
        midle = SearchDataItem(Point([0.5], ["B", "B"]), 3.5, discreteValueIndex=3)
        midle.globalR = 0.45
        midle.SetIndex(0)
        self.mixedIntegerMethod.searchData.InsertDataItem(midle, last)
        # 0.0 3.5 4.0
        new = SearchDataItem(Point([0.75], ["B", "B"]), 3.75, discreteValueIndex=3)
        new.globalR = 1.6
        self.mixedIntegerMethod.evolvent.GetImage = Mock(side_effect=self.mock_GetImage)

        get_new, get_old = self.mixedIntegerMethod.CalculateIterationPoint()
        self.assertEqual(new.GetX(), get_new.GetX())
        self.assertEqual(new.point.floatVariables[0], get_new.point.floatVariables[0])
        self.assertEqual(new.GetDiscreteValueIndex(), get_new.GetDiscreteValueIndex())
        self.assertEqual(last.point.floatVariables[0], get_old.point.floatVariables[0])
        self.assertEqual(last.GetX(), get_old.GetX())
        self.assertEqual(last.GetDiscreteValueIndex(), get_old.GetDiscreteValueIndex())
        self.assertEqual(2, self.mixedIntegerMethod.searchData.solution.numberOfGlobalTrials)
        mock_CalculateGlobalR.assert_called()
        self.mixedIntegerMethod.evolvent.GetImage.assert_called()


if __name__ == '__main__':
    unittest.main()
