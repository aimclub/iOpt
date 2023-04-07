from __future__ import annotations

import copy
from multiprocessing import Pool

from iOpt.method.icriterion_evaluate_method import ICriterionEvaluateMethod
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

import sys

sys.setrecursionlimit(10000)


class Calculator:
    pool: Pool = None
    evaluateMethod: ICriterionEvaluateMethod = None

    def __init__(self,
                 evaluateMethod: ICriterionEvaluateMethod,
                 parameters: SolverParameters
                 ):
        r"""
        Конструктор класса Calculator

        :param evaluateMethod: метод вычислений, проводящий поисковые испытания по заданным правилам.
        :param parameters: параметры решения задачи оптимизации.
        """
        self.evaluateMethod = evaluateMethod
        self.parameters = parameters
        Calculator.pool = Pool(parameters.numberOfParallelPoints,
                               initializer=Calculator.worker_init,
                               initargs=(self.evaluateMethod,))

    r"""
    Инициализация метода вычислений в каждем процессе из пула процессов Calculator.Pool

    :param evaluateMethod: метод вычислений, проводящий поисковые испытания по заданным правилам.
    """
    @staticmethod
    def worker_init(evaluateMethod: ICriterionEvaluateMethod):
        Calculator.evaluateMethod = evaluateMethod

    r"""
    Метод проведения испытаний в процессе из пула процессов Calculator.Pool

    :param point: точка проведения испытания
    """
    @staticmethod
    def worker(point: SearchDataItem) -> SearchDataItem:
        Calculator.evaluateMethod.CalculateFunctionals(point)
        return point

    r"""
    Метод проведения испытаний для множества точек

    :param points: точки проведения испытаний
    """

    def CalculateFunctionalsForItems(self, points: list[SearchDataItem]) -> list[SearchDataItem]:
        # Ниже реализация цикла через пулл процессов
        # for point in points:
        #     self.worker(point, self.method)

        points_copy = []
        for point in points:
            sd = SearchDataItem(y=copy.deepcopy(point.point), x=copy.deepcopy(point.GetX()),
                                functionValues=copy.deepcopy(point.functionValues),
                                discreteValueIndex=point.GetDiscreteValueIndex())
            points_copy.append(sd)

        points_res = Calculator.pool.map(Calculator.worker, points_copy)

        for point, point_r in zip(points, points_res):
            self.evaluateMethod.CopyFunctionals(point, point_r)

        return points
