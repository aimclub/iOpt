from __future__ import annotations

import copy
from itertools import repeat
from multiprocessing import Pool

from iOpt.method.imethod import IMethod
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

import sys
sys.setrecursionlimit(10000)

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
        Calculator.pool = Pool(parameters.numberOfParallelPoints,
                               initializer=Calculator.worker_init,
                               initargs=(self.method,))
        # Calculator.pool = Pool(1)

    @staticmethod
    def worker_init(method: IMethod):
        Calculator.method = method
        # print("Init !", flush=True)

    @staticmethod
    def worker(point: SearchDataItem) -> SearchDataItem:
        Calculator.method.CalculateFunctionals(point)
        return point

    def CalculateFunctionalsForItems(self, points: list[SearchDataItem]) -> list[SearchDataItem]:
        # for point in points:
        #     self.worker(point, self.method)

#        Calculator.pool = Pool(self.parameters.numberOfParallelPoints)
#         points_res = Calculator.pool.starmap(Calculator.worker, zip(points, repeat(self.method)))
        points_copy = []
        for point in points:
            sd = SearchDataItem(y=copy.deepcopy(point.point), x=copy.deepcopy(point.GetX()),
                 functionValues = copy.deepcopy(point.functionValues),
                 discreteValueIndex = point.GetDiscreteValueIndex())
            points_copy.append(sd)

        points_res = Calculator.pool.map(Calculator.worker, points_copy)
#        Calculator.pool.close()

        for point, point_r in zip(points, points_res):
            point.functionValues[0] = point_r.functionValues[0]
            point.SetZ(point_r.functionValues[0].value)
            point.SetIndex(0)

        return points

