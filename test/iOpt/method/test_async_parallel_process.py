import unittest
from time import sleep

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.async_parallel_process import AsyncParallelProcess
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.problem import Problem
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue


class ProblemForTest(Problem):
    def __init__(self):
        super().__init__()
        self.dimension = 1
        self.number_of_float_variables = 1
        self.number_of_constraints = 0
        self.number_of_objectives = 1
        self.lower_bound_of_float_variables = [-1]
        self.upper_bound_of_float_variables = [1]

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        sleep(1 - point.float_variables[0])
        function_value.value = point.float_variables[0] ** 2
        return function_value


class TestAsyncParallelProcess(unittest.TestCase):
    def setUp(self):
        param = SolverParameters(number_of_parallel_points=2, iters_limit=4)
        problem = ProblemForTest()
        task = OptimizationTask(problem)
        evolvent = Evolvent(
            problem.lower_bound_of_float_variables,
            problem.upper_bound_of_float_variables,
            problem.number_of_float_variables,
        )
        search_data = SearchData(problem)
        method = Method(param, task, evolvent, search_data)
        self.async_parallel_process = AsyncParallelProcess(
            param, task, evolvent, search_data, method, []
        )

    def test_Solve(self):
        self.async_parallel_process.solve()
        items = self.async_parallel_process.search_data.get_last_items(5)
        self.assertAlmostEqual(-1 / 3, items[1].get_y().float_variables[0])
        self.assertAlmostEqual(1 / 3, items[2].get_y().float_variables[0])
        self.assertAlmostEqual(2 / 3, items[3].get_y().float_variables[0])
        self.assertAlmostEqual(-2 / 3, items[4].get_y().float_variables[0])
        self.assertAlmostEqual(
            1 / 3,
            self.async_parallel_process.search_data.solution.best_trials[0]
            .point.float_variables[0],
        )
        self.assertAlmostEqual(
            1 / 9,
            self.async_parallel_process.search_data.solution.best_trials[0]
            .function_values[0].value,
        )
        self.assertEqual(4, self.async_parallel_process.method.iterations_count)


if __name__ == "__main__":
    unittest.main()
