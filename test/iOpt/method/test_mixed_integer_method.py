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
        mock_problem.number_of_float_variables = 1
        mock_problem.number_of_discrete_variables = 2
        mock_problem.discrete_variable_values = [["A", "B"], ["A", "B"]]
        mock_problem.lower_bound_of_float_variables = [[-0.2]]
        mock_problem.upper_bound_of_float_variables = [[2.0]]
        mock_problem.number_of_objectives = 1
        mock_problem.number_of_constraints = 0

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
        searchDataItem.set_z(searchDataItem.get_x() * 0.34)
        searchDataItem.set_index(0)
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
        self.mixedIntegerMethod.calculate_functionals = Mock(side_effect=self.mock_CalculateFunctionals)
        self.mixedIntegerMethod.evolvent.get_image = Mock(side_effect=self.mock_GetImage)
        self.mixedIntegerMethod.first_iteration()
        self.assertEqual(self.mixedIntegerMethod.search_data.get_count(), 9)
        self.mixedIntegerMethod.calculate_functionals.assert_called()
        self.mixedIntegerMethod.evolvent.get_image.assert_called()

    def test_FirstIterationParallel(self):
        self.mixedIntegerMethod.parameters.number_of_parallel_points = 5
        self.mixedIntegerMethod.evolvent.get_image = Mock(side_effect=self.mock_GetImage)
        calculator = Calculator(None, self.mixedIntegerMethod.parameters)
        calculator.calculate_functionals_for_items = Mock(side_effect=self.mock_CalculateFunctionalsForItems)
        self.mixedIntegerMethod.first_iteration(calculator)
        self.assertEqual(self.mixedIntegerMethod.search_data.get_count(), 13)
        calculator.calculate_functionals_for_items.assert_called()
        self.mixedIntegerMethod.evolvent.get_image.assert_called()

    @mock.patch('iOpt.method.index_method.IndexMethod.calculate_global_r')
    def test_CalculateIterationPoint(self, mock_calculate_global_r):
        first = SearchDataItem(Point([0.0], ["A", "A"]), 0.0, discrete_value_index=0)
        last = SearchDataItem(Point([1.0], ["B", "B"]), 4.0, discrete_value_index=3)
        self.mixedIntegerMethod.search_data.insert_first_data_item(first, last)
        midl = SearchDataItem(Point([0.5], ["B", "B"]), 3.5, discrete_value_index=3)
        midl.globalR = 0.45
        midl.set_index(0)
        self.mixedIntegerMethod.search_data.insert_data_item(midl, last)
        # 0.0 3.5 4.0
        new = SearchDataItem(Point([0.75], ["B", "B"]), 1.75, discrete_value_index=3)
        self.mixedIntegerMethod.evolvent.get_image = Mock(side_effect=self.mock_GetImage)

        self.mixedIntegerMethod.best = midl

        get_new, get_old = self.mixedIntegerMethod.calculate_iteration_point()
        self.assertEqual(new.get_x(), get_new.get_x())
        self.assertEqual(new.point.float_variables[0], get_new.point.float_variables[0])
        self.assertEqual(new.get_discrete_value_index(), get_new.get_discrete_value_index())
        self.assertEqual(midl.point.float_variables[0], get_old.point.float_variables[0])
        self.assertEqual(midl.get_x(), get_old.get_x())
        self.assertEqual(midl.get_discrete_value_index(), get_old.get_discrete_value_index())
        self.assertEqual(2, self.mixedIntegerMethod.search_data.solution.number_of_global_trials)
        mock_calculate_global_r.assert_called()
        self.mixedIntegerMethod.evolvent.get_image.assert_called()

    def test_RastriginInt_Solve_Dimension_3_Discrete_2_p_1(self):
        epsVal = 0.01
        r = 3.5
        problem = RastriginInt(3, 2)
        params = SolverParameters(r=r, eps=epsVal, number_of_parallel_points=4)
        solver = Solver(problem, parameters=params)

        sol = solver.solve()

        res = True
        for j in range(problem.dimension - problem.number_of_discrete_variables):
            print(sol.best_trials[0].point.float_variables[j])
            fabsx = np.abs(problem.known_optimum[0].point.float_variables[j] -
                           sol.best_trials[0].point.float_variables[j])
            fm = epsVal * (problem.upper_bound_of_float_variables[j] -
                           problem.lower_bound_of_float_variables[j])
            if fabsx > fm:
                res = False

        self.assertEqual(res, True)
        self.assertAlmostEqual(sol.number_of_global_trials, self.globalTrials_RI_d_3, delta=self.eps_trials_3)

    def test_RastriginInt_Solve_Dimension_3_Discrete_2_p_3(self):
        epsVal = 0.01
        r = 3.5
        problem = RastriginInt(3, 2)
        params = SolverParameters(r=r, eps=epsVal, number_of_parallel_points=3)
        solver = Solver(problem, parameters=params)

        sol = solver.solve()

        res = True
        for j in range(problem.dimension - problem.number_of_discrete_variables):
            fabsx = np.abs(problem.known_optimum[0].point.float_variables[j] -
                           sol.best_trials[0].point.float_variables[j])
            fm = epsVal * (problem.upper_bound_of_float_variables[j] -
                           problem.lower_bound_of_float_variables[j])
            if fabsx > fm:
                res = False

        self.assertEqual(res, True)
        self.assertAlmostEqual(sol.number_of_global_trials, self.globalTrials_RI_d_3, delta=self.eps_trials_3)

    def test_RastriginInt_Solve_Dimension_4_Discrete_2_p_1(self):
        epsVal = 0.01
        r = 3.5
        problem = RastriginInt(4, 2)
        params = SolverParameters(r=r, eps=epsVal, number_of_parallel_points=1)
        solver = Solver(problem, parameters=params)

        sol = solver.solve()

        res = True
        for j in range(problem.dimension - problem.number_of_discrete_variables):
            fabsx = np.abs(problem.known_optimum[0].point.float_variables[j] -
                           sol.best_trials[0].point.float_variables[j])
            fm = epsVal * (problem.upper_bound_of_float_variables[j] -
                           problem.lower_bound_of_float_variables[j])
            if fabsx > fm:
                res = False

        self.assertEqual(res, True)
        self.assertAlmostEqual(sol.number_of_global_trials, self.globalTrials_RI_d_4, delta=self.eps_trials_4)


if __name__ == '__main__':
    unittest.main()