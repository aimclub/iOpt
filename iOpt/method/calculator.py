from __future__ import annotations

from itertools import repeat
from multiprocessing import Pool

from sqlalchemy.dialects.mssql import IMAGE

from iOpt.method.imethod import IMethod
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters


class Calculator:

    def __init__(self,
                 method: IMethod,
                 parameters: SolverParameters
                 ):
        r"""
        Конструктор класса Calculator

        :param method: метод оптимизации, проводящий поисковые испытания по заданным правилам.
        :param parameters: параметры решения задачи оптимизации.
        """
        self.method = method
        self.parameters = parameters
        Calculator.pool = Pool(parameters.numberOfParallelPoints)
        # Calculator.pool = Pool(1)

    @staticmethod
    def worker(point: SearchDataItem, method: IMethod) -> SearchDataItem:
        method.CalculateFunctionals(point)
        return point

    def CalculateFunctionalsForItems(self, points: list[SearchDataItem]) -> list[SearchDataItem]:
        # for point in points:
        #     self.worker(point, self.method)

        points_res = Calculator.pool.starmap(Calculator.worker, zip(points, repeat(self.method)))

        for point, point_r in zip(points, points_res):
            point.functionValues[0] = point_r.functionValues[0]
            point.SetZ(point_r.functionValues[0].value)
            point.SetIndex(0)

        return points

