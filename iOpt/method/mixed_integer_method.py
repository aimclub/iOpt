from __future__ import annotations

import itertools

import copy
import math
from typing import Tuple

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

from iOpt.method.method import Method
from iOpt.method.index_method import IndexMethod
from iOpt.trial import Point
from iOpt.problem import Problem

class MixedIntegerMethod(Method):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        super(MixedIntegerMethod, self).__init__(parameters, task, evolvent, searchData)

        # u = {i, j, k}, i = {0, 1, 2}, j = {0, 1}, k = {0, 1, 2, 3, 4} -> 3*2*4=24

        list_discreteValues = list(task.problem.discreteVariableValues)
        self.discreteParameters = list(itertools.product(*list_discreteValues))
        # определяем количество сочетаний параметров
        self.numberOfParameterCombinations = len(self.discreteParameters)
        # 0 0.5 1  1.5 2   2.5  3    3.5 4

    def FirstIteration(self, calculator: Calculator = None) -> None:
        r"""
        Метод выполняет первую итерацию Алгоритма Глобального Поиска.
        """
        self.iterationsCount = 1
        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        left = SearchDataItem(Point(self.evolvent.GetImage(0.0), self.discreteParameters[0]), 0.0)
        image_right = self.evolvent.GetImage(1.0)
        right: list[SearchDataItem] = []

        # [(x + y - 1)/y]
        numberOfPointsInOneInterval =\
            int(math.modf((self.parameters.numberOfParallelPoints + self.numberOfParameterCombinations - 1)
                          / self.numberOfParameterCombinations)[1])

        h: float = 1.0 / (numberOfPointsInOneInterval + 1)
        items: list[SearchDataItem] = []
        image_x: list = []

        for id_comb in range(self.numberOfParameterCombinations):
            for i in range(numberOfPointsInOneInterval):
                x = (id_comb * numberOfPointsInOneInterval) + h * (i + 1)
                if id_comb == 0:
                    image_x.append(self.evolvent.GetImage(x))

                y = Point(copy.copy(image_x[i]), self.discreteParameters[id_comb])
                item = SearchDataItem(y, x, discreteValueIndex=id_comb)
                items.append(item)

            right.append(SearchDataItem(Point(copy.copy(image_right), self.discreteParameters[id_comb]),
                                        float(id_comb + 1), discreteValueIndex=id_comb))
        if calculator is None:
            for item in items:
                self.CalculateFunctionals(item)
        else:
            calculator.CalculateFunctionalsForItems(items)

        for item in items:
            self.UpdateOptimum(item)

        left.delta = 0
        # left надо для всех считать
        self.CalculateGlobalR(left, None)

        items[0].delta = self.CalculateDelta(left, items[0], self.dimension)

        self.CalculateGlobalR(items[0], left)
        for id_comb in range(self.numberOfParameterCombinations):
            if id_comb > 0:
                # вычисление left
                index = id_comb*numberOfPointsInOneInterval
                items[index].delta = self.CalculateDelta(right[id_comb-1], items[index], self.dimension)
                self.CalculateGlobalR(items[index], right[id_comb-1])

            for id_item in range(1, numberOfPointsInOneInterval):
                index = id_comb * numberOfPointsInOneInterval + id_item
                items[index].delta = self.CalculateDelta(items[index- 1], items[index], self.dimension)
                self.CalculateGlobalR(items[index], items[index - 1])
                self.CalculateM(items[index], items[index - 1])

            left_index = id_comb * numberOfPointsInOneInterval + numberOfPointsInOneInterval - 1
            right[id_comb].delta = self.CalculateDelta(items[left_index], right[id_comb], self.dimension)
            self.CalculateGlobalR(right[id_comb], items[left_index])


        # вставить left  и right, потом middle
        self.searchData.InsertFirstDataItem(left, right[-1])

        for right_item in range(self.numberOfParameterCombinations):
            if right_item < self.numberOfParameterCombinations - 1:
                self.searchData.InsertDataItem(right[right_item], right[-1])

            for id_item in range(numberOfPointsInOneInterval):
                index = right_item*numberOfPointsInOneInterval + id_item
                self.searchData.InsertDataItem(items[index], right[right_item])

        self.recalcR = True
        self.recalcM = True


    def CalculateIterationPoint(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}`.

        :return: :math:`x^{k+1}` - точка нового испытания, и :math:`x_t` - левая точка интервала :math:`[x_{t-1},x_t]`,
          которому принадлежит :math:`x^{k+1}`, т.е. :math:`x^{k+1} \in [x_{t-1},x_t]`.
        """
        if self.recalcM is True:
            self.RecalcM()
        if self.recalcR is True:
            self.RecalcAllCharacteristics()

        old = self.searchData.GetDataItemWithMaxGlobalR()
        self.min_delta = min(old.delta, self.min_delta)
        newx = self.CalculateNextPointCoordinate(old)
        newy = self.evolvent.GetImage(newx - math.modf(newx)[1])
        new = copy.deepcopy(SearchDataItem(Point(newy, old.point.discreteVariables),
                                           newx, discreteValueIndex=old.GetDiscreteValueIndex()))
        # Обновление числа испытаний
        self.searchData.solution.numberOfGlobalTrials += 1

        return new, old

    @staticmethod
    def GetDiscreteParameters(problem: Problem) -> list:
        list_discreteValues = list(problem.discreteVariableValues)
        return list(itertools.product(*list_discreteValues))
 