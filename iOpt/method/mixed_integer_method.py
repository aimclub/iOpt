from __future__ import annotations

import itertools

import copy
import math
from typing import Tuple

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

from iOpt.method.index_method import IndexMethod
from iOpt.trial import Point, FunctionValue
from iOpt.problem import Problem


class MixedIntegerMethod(IndexMethod):
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
        left = SearchDataItem(Point(self.evolvent.GetImage(0.0), self.discreteParameters[0]), 0.0,
                              functionValues=[FunctionValue()] * self.numberOfAllFunctions)
        image_right = self.evolvent.GetImage(1.0)
        right: list[SearchDataItem] = []

        # [(x + y - 1)/y]

        items: list[SearchDataItem] = []
        image_x: list = []
        is_init_image_x: bool = False

        numberOfPointsInOneInterval = \
            int(math.modf((self.parameters.numberOfParallelPoints + self.numberOfParameterCombinations - 1)
                          / self.numberOfParameterCombinations)[1])

        h: float = 1.0 / (numberOfPointsInOneInterval + 1)

        if self.parameters.startPoint:

            for id_comb in range(self.numberOfParameterCombinations):

                if np.array_equal(self.parameters.startPoint.discreteVariables, self.discreteParameters[id_comb]):
                    numTemp = numberOfPointsInOneInterval - 1

                    yStartPoint = Point(copy.copy(self.parameters.startPoint.floatVariables),
                                        self.discreteParameters[id_comb])
                    xStartPoint = id_comb + self.evolvent.GetInverseImage(self.parameters.startPoint.floatVariables)
                    itemStartPoint = SearchDataItem(yStartPoint, xStartPoint, discreteValueIndex=id_comb,
                                          functionValues=[FunctionValue()] * self.numberOfAllFunctions)

                    isAddStartPoint: bool = False

                    for i in range(numTemp):
                        x = id_comb + h * (i + 1)

                        y_temp = self.evolvent.GetImage(x)

                        y = Point(copy.copy(y_temp), self.discreteParameters[id_comb])
                        item = SearchDataItem(y, x, discreteValueIndex=id_comb,
                                              functionValues=[FunctionValue()] * self.numberOfAllFunctions)
                        if x < xStartPoint < id_comb + h * (i + 1):
                            items.append(item)
                            items.append(itemStartPoint)
                            isAddStartPoint = True
                        else:
                            items.append(item)

                    if not isAddStartPoint:
                        items.append(itemStartPoint)

                else:

                    for i in range(numberOfPointsInOneInterval):
                        x = id_comb + h * (i + 1)
                        if not is_init_image_x:
                            image_x.append(self.evolvent.GetImage(x))

                        y = Point(copy.copy(image_x[i]), self.discreteParameters[id_comb])
                        item = SearchDataItem(y, x, discreteValueIndex=id_comb,
                                              functionValues=[FunctionValue()] * self.numberOfAllFunctions)
                        items.append(item)

                right.append(SearchDataItem(Point(copy.copy(image_right), self.discreteParameters[id_comb]),
                                            float(id_comb + 1),
                                            functionValues=[FunctionValue()] * self.numberOfAllFunctions,
                                            discreteValueIndex=id_comb))

                if not is_init_image_x:
                    is_init_image_x = True
        else:
            for id_comb in range(self.numberOfParameterCombinations):
                for i in range(numberOfPointsInOneInterval):
                    x = (id_comb * numberOfPointsInOneInterval) + h * (i + 1)
                    if not is_init_image_x:
                        image_x.append(self.evolvent.GetImage(x))

                    y = Point(copy.copy(image_x[i]), self.discreteParameters[id_comb])
                    item = SearchDataItem(y, x, discreteValueIndex=id_comb,
                                          functionValues=[FunctionValue()] * self.numberOfAllFunctions)
                    items.append(item)

                right.append(SearchDataItem(Point(copy.copy(image_right), self.discreteParameters[id_comb]),
                                            float(id_comb + 1),
                                            functionValues=[FunctionValue()] * self.numberOfAllFunctions,
                                            discreteValueIndex=id_comb))

                if not is_init_image_x:
                    is_init_image_x = True

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
                index = id_comb * numberOfPointsInOneInterval
                items[index].delta = self.CalculateDelta(right[id_comb - 1], items[index], self.dimension)
                self.CalculateGlobalR(items[index], right[id_comb - 1])

            for id_item in range(1, numberOfPointsInOneInterval):
                index = id_comb * numberOfPointsInOneInterval + id_item
                items[index].delta = self.CalculateDelta(items[index - 1], items[index], self.dimension)
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
                index = right_item * numberOfPointsInOneInterval + id_item
                self.searchData.InsertDataItem(items[index], right[right_item])

        self.recalcR = True
        self.recalcM = True

        self.iterationsCount = len(items)

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
                                           newx, discreteValueIndex=old.GetDiscreteValueIndex(),
                                           functionValues=[FunctionValue()] * self.numberOfAllFunctions))
        # Обновление числа испытаний
        self.searchData.solution.numberOfGlobalTrials += 1

        return new, old

    @staticmethod
    def GetDiscreteParameters(problem: Problem) -> list:
        list_discreteValues = list(problem.discreteVariableValues)
        return list(itertools.product(*list_discreteValues))

    def CalculateM(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление оценки константы Гельдера между между curr_point и left_point.

        :param curr_point: правая точка интервала
        :param left_point: левая точка интервала
        """
        # Обратить внимание на вычисление расстояния, должен использоваться метод CalculateDelta
        if curr_point is None:
            print("CalculateM: curr_point is None")
            raise RuntimeError("CalculateM: curr_point is None")
        if left_point is None:
            return
        index = curr_point.GetIndex()
        if index < 0:
            return
        m = 0.0
        if left_point.GetIndex() == index:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left_point.GetZ() - curr_point.GetZ()) / curr_point.delta
        else:
            # Ищем слева
            other_point = left_point
            while (other_point is not None) and (other_point.GetIndex() < curr_point.GetIndex()):
                if other_point.GetDiscreteValueIndex() == curr_point.GetDiscreteValueIndex():
                    other_point = other_point.GetLeft()
                else:
                    other_point = None
                    break
            if other_point is not None and other_point.GetIndex() >= 0:
                # print(index)
                m = abs(other_point.functionValues[index].value - curr_point.GetZ()) / \
                    self.CalculateDelta(other_point, curr_point, self.dimension)

            # Ищем справа
            other_point = left_point.GetRight()
            if other_point is not None and other_point is curr_point:  # возможно только при пересчёте M
                other_point = other_point.GetRight()
            while (other_point is not None) and (other_point.GetIndex() < curr_point.GetIndex()):
                if other_point.GetDiscreteValueIndex() == curr_point.GetDiscreteValueIndex():
                    other_point = other_point.GetRight()
                else:
                    other_point = None
                    break

            if other_point is not None and other_point.GetIndex() >= 0:
                m = max(m, abs(curr_point.GetZ() - other_point.functionValues[index].value) / \
                        self.CalculateDelta(curr_point, other_point, self.dimension))

        if m > self.M[index] or (self.M[index] == 1.0 and m > 1e-12):
            self.M[index] = m
            self.recalcR = True
