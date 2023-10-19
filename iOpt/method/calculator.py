from __future__ import annotations

import copy
from pathos.multiprocessing import _ProcessPool

from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

import sys

# возможно стоит удалить
sys.setrecursionlimit(10000)


class Calculator:
    pool: _ProcessPool = None
    evaluate_method: ICriterionEvaluateMethod = None

    def __init__(self,
                 evaluate_method: ICriterionEvaluateMethod,
                 parameters: SolverParameters
                 ):
        r"""
        Constructor of class Calculator

        :param evaluate_method: a computational method that performs search trials according to specified rules.
        :param parameters: solution parameters of the optimization problem.
        """
        self.evaluate_method = evaluate_method
        self.parameters = parameters
        Calculator.worker_init(self.evaluate_method)
        Calculator.pool = _ProcessPool(parameters.number_of_parallel_points,
                                       initializer=Calculator.worker_init,
                                       initargs=(self.evaluate_method,))

    r"""
    Initialize the calculation method in each process from the process pool Calculator.Pool

    :param evaluate_method: a computational method that performs search trials according to specified rules.
    """

    @staticmethod
    def worker_init(evaluate_method: ICriterionEvaluateMethod):
        Calculator.evaluate_method = evaluate_method

    r"""
    Сalculation method in each process from the process pool Calculator.Pool

    :param point: trial point.
    """

    @staticmethod
    def worker(point: SearchDataItem) -> SearchDataItem:
        try:
            Calculator.evaluate_method.calculate_functionals(point)
        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-10)
        return point

    r"""
    Сalculation method for multiple points

    :param points: trial points.
    """

    def calculate_functionals_for_items(self, points: list[SearchDataItem]) -> list[SearchDataItem]:
        # пока оставленно на случай отладки
        # for point in points:
        #     self.worker(point, self.method)

        # Ниже реализация цикла через пулл процессов
        points_copy = []
        for point in points:
            sd = SearchDataItem(y=copy.deepcopy(point.point), x=copy.deepcopy(point.get_x()),
                                function_values=copy.deepcopy(point.function_values),
                                discrete_value_index=point.get_discrete_value_index())
            points_copy.append(sd)

        points_res = Calculator.pool.map(Calculator.worker, points_copy)

        for point, point_r in zip(points, points_res):
            self.evaluate_method.copy_functionals(point, point_r)

        return points

    def __del__(self):
        self.pool.close()
