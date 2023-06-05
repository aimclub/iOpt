from __future__ import annotations

import math
import sys

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.method import Method
from iOpt.trial import FunctionValue, FunctionType


class IndexMethod(Method):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        super(IndexMethod, self).__init__(parameters, task, evolvent, searchData)

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        try:
            number_of_constraints = self.task.problem.numberOfConstraints
            for i in range(number_of_constraints):
                point.functionValues[i] = FunctionValue(FunctionType.CONSTRAINT, i)  # ???
                point = self.task.Calculate(point, i)
                point.SetZ(point.functionValues[i].value)
                point.SetIndex(i)
                if point.GetZ() > 0:
                    return point
            point.functionValues[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV, number_of_constraints)
            point = self.task.Calculate(point, number_of_constraints)
            point.SetZ(point.functionValues[number_of_constraints].value)
            point.SetIndex(number_of_constraints)
        except Exception:
            point.SetZ(sys.float_info.max)
            point.SetIndex(-10)


        return point

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
                other_point = other_point.GetLeft()
            if other_point is not None and other_point.GetIndex() >= 0:
                # print(index)
                m = abs(other_point.functionValues[index].value - curr_point.GetZ()) / \
                    self.CalculateDelta(other_point, curr_point, self.dimension)
            # Ищем справа
            other_point = left_point.GetRight()
            if other_point is not None and other_point is curr_point:  # возможно только при пересчёте M
                other_point = other_point.GetRight()
            while (other_point is not None) and (other_point.GetIndex() < curr_point.GetIndex()):
                other_point = other_point.GetRight()
            if other_point is not None and other_point.GetIndex() >= 0:
                m = max(m, abs(curr_point.GetZ() - other_point.functionValues[index].value) / \
                        self.CalculateDelta(curr_point, other_point, self.dimension))

        if m > self.M[index] or (self.M[index] == 1.0 and m > 1e-12):
            self.M[index] = m
            self.recalcR = True

    def CalculateGlobalR(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление глобальной характеристики интервала [left_point, curr_point].

        :param curr_point: правая точка интервала.
        :param left_point: левая точка интервала.
        """

        # Сюда переедет целиком CalculateGlobalR из Method, а там останется только случай с равными индексами
        if curr_point is None:
            print("CalculateGlobalR: Curr point is NONE")
            raise Exception("CalculateGlobalR: Curr point is NONE")
        if left_point is None:
            curr_point.globalR = -np.infty
            return None
        zl = left_point.GetZ()
        zr = curr_point.GetZ()
        r = self.parameters.r
        deltax = curr_point.delta

        if left_point.GetIndex() < 0 and curr_point.GetIndex() < 0:
            globalR = 2 * deltax - 4 * math.fabs(self.Z[0]) / (r * self.M[0])
        elif left_point.GetIndex() == curr_point.GetIndex():
            v = curr_point.GetIndex()
            globalR = deltax + (zr - zl) * (zr - zl) / (deltax * self.M[v] * self.M[v] * r * r) - \
                      2 * (zr + zl - 2 * self.Z[v]) / (r * self.M[v])
        elif left_point.GetIndex() < curr_point.GetIndex():
            v = curr_point.GetIndex()
            globalR = 2 * deltax - 4 * (zr - self.Z[v]) / (r * self.M[v])
        else:
            v = left_point.GetIndex()
            globalR = 2 * deltax - 4 * (zl - self.Z[v]) / (r * self.M[v])
        curr_point.globalR = globalR

    def UpdateZ(self, point: SearchDataItem) -> None:
        for i in range(point.GetIndex()):
            if self.Z[i] > point.functionValues[i].value:
                self.Z[i] = point.functionValues[i].value
                self.recalcR = True

    def RecalcAllCharacteristics(self) -> None:
        for i in range(self.best.GetIndex()):
            self.Z[i] = -self.M[i] * self.parameters.epsR
        self.Z[self.best.GetIndex()] = self.best.GetZ()
        super().RecalcAllCharacteristics()

    def UpdateOptimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.

        :param point: точка нового испытания.
        """

        if self.best is None or self.best.GetIndex() < point.GetIndex() or (
                self.best.GetIndex() == point.GetIndex() and point.GetZ() < self.best.GetZ()):
            self.best = point
            self.recalcR = True
            self.Z[point.GetIndex()] = point.GetZ()
        # self.UpdateZ(point)
        self.searchData.solution.bestTrials[0] = self.best
