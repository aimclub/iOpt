from __future__ import annotations
from typing import List

import numpy as np
import itertools

import copy
import math
from typing import Tuple

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters

from iOpt.method.method import Method
from iOpt.method.index_method import IndexMethod
from iOpt.trial import Point



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


        numberOfDisreteVariables = task.problem.numberOfDisreteVariables

        # u = {i, j, k}, i = {0, 1, 2}, j = {0, 1}, k = {0, 1, 2, 3, 4} -> 3*2*4=24

        # numberOfParameterCombinations = 1
        # for i in range(numberOfDisreteVariables):
        #     numberOfParameterCombinations *= len(task.problem.discreteVariableValues[i])

        list_discreteValues = list(task.problem.discreteVariableValues)
        self.discreteParameters = list(itertools.product(*list_discreteValues))
        # определяем количество сочетаний параметров
        self.numberOfParameterCombinations = len(self.discreteParameters)
        # 0 0.5 1  1.5 2   2.5  3    3.5 4

    def FirstIteration(self) -> None:
        r"""
        Метод выполняет первую итерацию Алгоритма Глобального Поиска.
        """
        # [0, 0.5, 1]
        self.iterationsCount = 1
        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        x: float = 0.5
        middle_image = self.evolvent.GetImage(x)
        left_image = self.evolvent.GetImage(0.0)
        right_image = self.evolvent.GetImage(1.0)

        y = Point(middle_image, self.discreteParameters[0])
        middle = SearchDataItem(y, x, discreteValueIndex=0)
        left = SearchDataItem(Point(left_image, self.discreteParameters[0]), 0.0, discreteValueIndex=0)
        left.SetIndex(-3)  # по умолчанию -2
        right = SearchDataItem(Point(right_image, self.discreteParameters[0]), 1.0, discreteValueIndex=0)
        right.SetIndex(-3)
        left.delta = 0
        middle.delta = self.CalculateDelta(left, middle, self.dimension)
        right.delta = self.CalculateDelta(middle, right, self.dimension)

        # Вычисление значения функции в 0.5
        self.CalculateFunctionals(middle)
        self.UpdateOptimum(middle)

        # Вычисление характеристик
        self.CalculateGlobalR(left, None)
        self.CalculateGlobalR(middle, left)
        self.CalculateGlobalR(right, middle)

        # вставить left  и right, потом middle
        self.searchData.InsertFirstDataItem(left, right)
        self.searchData.InsertDataItem(middle, right)

        for i in range(1, self.numberOfParameterCombinations):
            # 1  2  3 ... self.numberOfParameterCombinations
            x = i + 0.5  # (2 * i + 1) / 2
            y = Point(middle_image, self.discreteParameters[i])
            middle = SearchDataItem(y, x, discreteValueIndex=i)
            left = self.searchData.GetLastItem().GetRight()
            right = SearchDataItem(Point(right_image, None), float(i + 1), discreteValueIndex=i)
            # index = - 3
            right.SetIndex(-3)
            middle.delta = self.CalculateDelta(left, middle, self.dimension)
            right.delta = self.CalculateDelta(middle, right, self.dimension)

            # Вычисление значения функции в 0.5
            self.CalculateFunctionals(middle)
            self.UpdateOptimum(middle)

            # Вычисление характеристик
           # self.CalculateGlobalR(left, self.searchData.GetLastItem())
            self.CalculateGlobalR(middle, left)
            self.CalculateGlobalR(right, middle)
            # addRightPoint
            self.searchData.InsertDataItem(right)
            self.searchData.InsertDataItem(middle, right)


    def CalculateDelta(lPoint: SearchDataItem, rPoint: SearchDataItem, dimension: int) -> float:
        """
        Вычисляет гельдерово расстояние в метрике Гельдера между двумя точками на отрезке [0,1],
          полученными при редукции размерности.

        :param lx: левая точка
        :param rx: правая точка
        :param dimension: размерность исходного пространства

        :return: гельдерово расстояние между lx и rx.
        """
        # Учесть что у левой точки может быть x = 1 и отрицательный индекс, тогда считать что x = 0
        pass

    def CalculateNextPointCoordinate(self, point: SearchDataItem) -> float:
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}` в заданном интервале :math:`[x_{t-1},x_t]`.

        :param point: интервал, заданный его правой точкой :math:`x_t`.

        :return: точка нового испытания :math:`x^{k+1}` в этом интервале.
        """

        # 0 0.5 1  1.5 2   2.5  3    3.5 4
        left = point.GetLeft()
        if left is None:
            print("CalculateNextPointCoordinate: Left point is NONE")
            raise Exception("CalculateNextPointCoordinate: Left point is NONE")
        xl = left.GetX()
        xl -= math.modf(xl)[1]
        xr = point.GetX()
        xr -= math.modf(xr)[1]
        idl = left.GetIndex()
        idr = point.GetIndex()
        if idl == idr:
            v = idr
            dif = point.GetZ() - left.GetZ()
            dg = -1.0
            if dif > 0:
                dg = 1.0

            x = 0.5 * (xl + xr)
            x -= 0.5 * dg * pow(abs(dif) / self.M[v], self.task.problem.numberOfFloatVariables) / self.parameters.r

        else:
            x = 0.5 * (xl + xr)
        if x <= xl or x >= xr:
            print(f"CalculateNextPointCoordinate: x is outside of interval {x} {xl} {xr}")
            raise Exception("CalculateNextPointCoordinate: x is outside of interval")
        return x

    def CalculateIterationPoint(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}`.

        :return: :math:`x^{k+1}` - точка нового испытания, и :math:`x_t` - левая точка интервала :math:`[x_{t-1},x_t]`,
          которому принадлежит :math:`x^{k+1}`, т.е. :math:`x^{k+1} \in [x_{t-1},x_t]`.
        """
        if self.recalc is True:
            self.RecalcAllCharacteristics()

        old = self.searchData.GetDataItemWithMaxGlobalR()
        self.min_delta = min(old.delta, self.min_delta)
        newx = self.CalculateNextPointCoordinate(old)
        newy = self.evolvent.GetImage(newx)
        new = copy.deepcopy(SearchDataItem(Point(newy, old.point.discreteVariables),
                                           newx, discreteValueIndex=old.GetDiscreteValueIndex()))
        return new, old