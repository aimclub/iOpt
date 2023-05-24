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
from problems.rastriginInt import RastriginInt
from iOpt.solver import Solver
import numpy as np

class TestMixedIntegerMethod(unittest.TestCase):

    @mock.patch('iOpt.problem.Problem')
    @mock.patch('iOpt.evolvent.evolvent')
    def setUp(self, mock_evolvent, mock_problem):
        mock_problem.numberOfFloatVariables = 1
        mock_problem.numberOfDiscreteVariables = 2
        mock_problem.discreteVariableValues = [["A", "B"], ["A", "B"]]
        mock_problem.lowerBoundOfFloatVariables = [[-0.2]]
        mock_problem.upperBoundOfFloatVariables = [[2.0]]
        mock_problem.numberOfObjectives = 1
        mock_problem.numberOfConstraints = 0

        task = OptimizationTask(mock_problem)
        self.mixedIntegerMethod = MixedIntegerMethod(SolverParameters(), task,
                                                     mock_evolvent, SearchData(mock_problem))
        self.globalTrials_RI_d_3 = 132
        self.globalTrials_RI_d_4 = 3619
        self.eps_trials_3 = 15
        self.eps_trials_4 = 35

    @staticmethod
    def mock_GetImage(x):
        return [x]

    @staticmethod
    def mock_CalculateFunctionals(searchDataItem: SearchDataItem):
        searchDataItem.SetZ(searchDataItem.GetX() * 0.34)
        searchDataItem.SetIndex(0)
        return searchDataItem

    @staticmethod
    def mock_CalculateFunctionalsForItems(items: List[SearchDataItem]):
        for item in items:
            TestMixedIntegerMethod.mock_CalculateFunctionals(item)
        return items

    def test_GetDiscreteParameters(self):
        dp = self.mixedIntegerMethod.GetDiscreteParameters(self.mixedIntegerMethod.task.problem)
        self.assertEqual(dp, [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")])

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

    @mock.patch('iOpt.method.index_method.IndexMethod.CalculateGlobalR')
    def test_CalculateIterationPoint(self, mock_CalculateGlobalR):
        first = SearchDataItem(Point([0.0], ["A", "A"]), 0.0, discreteValueIndex=0)
        last = SearchDataItem(Point([1.0], ["B", "B"]), 4.0, discreteValueIndex=3)
        self.mixedIntegerMethod.searchData.InsertFirstDataItem(first, last)
        midl = SearchDataItem(Point([0.5], ["B", "B"]), 3.5, discreteValueIndex=3)
        midl.globalR = 0.45
        midl.SetIndex(0)
        self.mixedIntegerMethod.searchData.InsertDataItem(midl, last)
        # 0.0 3.5 4.0
        new = SearchDataItem(Point([0.75], ["B", "B"]), 1.75, discreteValueIndex=3)
        self.mixedIntegerMethod.evolvent.GetImage = Mock(side_effect=self.mock_GetImage)

        self.mixedIntegerMethod.best = midl

        get_new, get_old = self.mixedIntegerMethod.CalculateIterationPoint()
        self.assertEqual(new.GetX(), get_new.GetX())
        self.assertEqual(new.point.floatVariables[0], get_new.point.floatVariables[0])
        self.assertEqual(new.GetDiscreteValueIndex(), get_new.GetDiscreteValueIndex())
        self.assertEqual(midl.point.floatVariables[0], get_old.point.floatVariables[0])
        self.assertEqual(midl.GetX(), get_old.GetX())
        self.assertEqual(midl.GetDiscreteValueIndex(), get_old.GetDiscreteValueIndex())
        self.assertEqual(2, self.mixedIntegerMethod.searchData.solution.numberOfGlobalTrials)
        mock_CalculateGlobalR.assert_called()
        self.mixedIntegerMethod.evolvent.GetImage.assert_called()

    def test_RastriginInt_Solve_Dimension_3_Discrete_2_p_1(self):
        epsVal = 0.01
        r = 3.5
        problem = RastriginInt(3, 2)
        params = SolverParameters(r=r, eps=epsVal, numberOfParallelPoints=4)
        solver = Solver(problem, parameters=params)

        sol = solver.Solve()

        res = True
        for j in range(problem.dimension - problem.numberOfDiscreteVariables):
            print(sol.bestTrials[0].point.floatVariables[j])
            fabsx = np.abs(problem.knownOptimum[0].point.floatVariables[j] -
                           sol.bestTrials[0].point.floatVariables[j])
            fm = epsVal * (problem.upperBoundOfFloatVariables[j] -
                           problem.lowerBoundOfFloatVariables[j])
            if fabsx > fm:
                res = False

        self.assertEqual(res, True)
        self.assertAlmostEqual(sol.numberOfGlobalTrials, self.globalTrials_RI_d_3, delta=self.eps_trials_3)

    def test_RastriginInt_Solve_Dimension_3_Discrete_2_p_3(self):
        epsVal = 0.01
        r = 3.5
        problem = RastriginInt(3, 2)
        params = SolverParameters(r=r, eps=epsVal, numberOfParallelPoints=3)
        solver = Solver(problem, parameters=params)

        sol = solver.Solve()

        res = True
        for j in range(problem.dimension - problem.numberOfDiscreteVariables):
            fabsx = np.abs(problem.knownOptimum[0].point.floatVariables[j] -
                           sol.bestTrials[0].point.floatVariables[j])
            fm = epsVal * (problem.upperBoundOfFloatVariables[j] -
                           problem.lowerBoundOfFloatVariables[j])
            if fabsx > fm:
                res = False

        self.assertEqual(res, True)
        self.assertAlmostEqual(sol.numberOfGlobalTrials, self.globalTrials_RI_d_3, delta=self.eps_trials_3)

    def test_RastriginInt_Solve_Dimension_4_Discrete_2_p_1(self):
        epsVal = 0.01
        r = 3.5
        problem = RastriginInt(4, 2)
        params = SolverParameters(r=r, eps=epsVal, numberOfParallelPoints=1)
        solver = Solver(problem, parameters=params)

        sol = solver.Solve()

        res = True
        for j in range(problem.dimension - problem.numberOfDiscreteVariables):
            fabsx = np.abs(problem.knownOptimum[0].point.floatVariables[j] -
                           sol.bestTrials[0].point.floatVariables[j])
            fm = epsVal * (problem.upperBoundOfFloatVariables[j] -
                           problem.lowerBoundOfFloatVariables[j])
            if fabsx > fm:
                res = False

        self.assertEqual(res, True)
        self.assertAlmostEqual(sol.numberOfGlobalTrials, self.globalTrials_RI_d_4, delta=self.eps_trials_4)


if __name__ == '__main__':
    unittest.main()