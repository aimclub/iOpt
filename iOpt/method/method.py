from __future__ import annotations

import copy
import math
import sys
from typing import Tuple

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue


class Method:
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        r"""
        Конструктор класса Method

        :param parameters: параметры решения задачи оптимизации.
        :param task: обёртка решаемой задачи.
        :param evolvent: развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param searchData: структура данных для хранения накопленной поисковой информации.
        """
        self.stop: bool = False
        self.recalcR: bool = True
        self.recalcM: bool = True
        self.iterationsCount: int = 0
        self.best: SearchDataItem = None

        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
        # change to np.array, but indexing np is slower
        self.M = [1.0 for _ in range(task.problem.numberOfObjectives + task.problem.numberOfConstraints)]
        self.Z = [np.infty for _ in range(task.problem.numberOfObjectives + task.problem.numberOfConstraints)]
        self.dimension = task.problem.numberOfFloatVariables  # А ДЛЯ ДИСКРЕТНЫХ?
        self.searchData.solution.solutionAccuracy = np.infty
        self.numberOfAllFunctions = task.problem.numberOfObjectives + task.problem.numberOfConstraints

    @property
    def min_delta(self):
        return self.searchData.solution.solutionAccuracy

    @min_delta.setter
    def min_delta(self, val):
        self.searchData.solution.solutionAccuracy = val

    # @staticmethod
    # def CalculateDelta(lx: float, rx: float, dimension: int) -> float:
    #     """
    #     Вычисляет гельдерово расстояние в метрике Гельдера между двумя точками на отрезке [0,1],
    #       полученными при редукции размерности.
    #
    #     :param lx: левая точка
    #     :param rx: правая точка
    #     :param dimension: размерность исходного пространства
    #
    #     :return: гельдерово расстояние между lx и rx.
    #     """
    #     return pow(rx - lx, 1.0 / dimension)

    def CalculateDelta(self, lPoint: SearchDataItem, rPoint: SearchDataItem, dimension: int) -> float:
        """
        Вычисляет гельдерово расстояние в метрике Гельдера между двумя точками на отрезке [0,1],
          полученными при редукции размерности.

        :param lPoint: левая точка
        :param rPoint: правая точка
        :param dimension: размерность исходного пространства

        :return: гельдерово расстояние между lx и rx.
        """
        return pow(rPoint.GetX() - lPoint.GetX(), 1.0 / dimension)

    def FirstIteration(self, calculator: Calculator = None) -> list[SearchDataItem]:
        r"""
        Метод выполняет первую итерацию Алгоритма Глобального Поиска.
        """

        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        left = SearchDataItem(Point(self.evolvent.GetImage(0.0), None), 0.,
                              functionValues=[FunctionValue()] * self.numberOfAllFunctions)
        right = SearchDataItem(Point(self.evolvent.GetImage(1.0), None), 1.0,
                               functionValues=[FunctionValue()] * self.numberOfAllFunctions)

        items: list[SearchDataItem] = []

        if self.parameters.startPoint:
            numberOfPoint: int = self.parameters.numberOfParallelPoints - 1
            h: float = 1.0 / (numberOfPoint + 1)

            yStartPoint = Point(copy.copy(self.parameters.startPoint.floatVariables), None)
            xStartPoint = self.evolvent.GetInverseImage(self.parameters.startPoint.floatVariables)

            itemStartPoint = SearchDataItem(yStartPoint, xStartPoint,
                                  functionValues=[FunctionValue()] * self.numberOfAllFunctions)

            isAddStartPoint: bool = False

            for i in range(numberOfPoint):
                x = h * (i + 1)
                y = Point(self.evolvent.GetImage(x), None)
                item = SearchDataItem(y, x,
                                      functionValues=[FunctionValue()] * self.numberOfAllFunctions)
                if x < xStartPoint < h * (i + 2):
                    items.append(item)
                    items.append(itemStartPoint)
                    isAddStartPoint = True
                else:
                    items.append(item)

            if not isAddStartPoint:
                items.append(itemStartPoint)
        else:

            numberOfPoint: int = self.parameters.numberOfParallelPoints
            h: float = 1.0 / (numberOfPoint + 1)

            for i in range(numberOfPoint):
                x = h * (i + 1)
                y = Point(self.evolvent.GetImage(x), None)
                item = SearchDataItem(y, x,
                                      functionValues=[FunctionValue()] * self.numberOfAllFunctions)
                items.append(item)

        if calculator is None:
            for item in items:
                self.CalculateFunctionals(item)
        else:
            calculator.CalculateFunctionalsForItems(items)

        for item in items:
            self.UpdateOptimum(item)

        left.delta = 0
        self.CalculateGlobalR(left, None)

        items[0].delta = self.CalculateDelta(left, items[0], self.dimension)
        self.CalculateGlobalR(items[0], left)
        for id_item, item in enumerate(items):
            if id_item > 0:
                items[id_item].delta = self.CalculateDelta(items[id_item - 1], items[id_item], self.dimension)
                self.CalculateGlobalR(items[id_item], items[id_item - 1])
                self.CalculateM(items[id_item], items[id_item - 1])

        right.delta = self.CalculateDelta(items[-1], right, self.dimension)
        self.CalculateGlobalR(right, items[-1])

        # вставить left  и right, потом middle
        self.searchData.InsertFirstDataItem(left, right)
        # self.searchData.InsertDataItem(middle, right)

        for item in items:
            self.searchData.InsertDataItem(item, right)

        self.recalcR = True
        self.recalcM = True

        self.iterationsCount = len(items)
        self.searchData.solution.numberOfGlobalTrials = len(items)

        return items

    def CheckStopCondition(self) -> bool:
        r"""
        Проверка условия остановки.
        Алгоритм должен завершить работу, когда достигнута точность eps или превышен лимит итераций.

        :return: True, если выполнен критерий остановки; False - в противном случае.
        """
        if self.min_delta < self.parameters.eps or self.iterationsCount >= self.parameters.globalMethodIterationCount:
            self.stop = True
        else:
            self.stop = False

        return self.stop

    def RecalcM(self) -> None:
        r"""
        Пересчёт оценки константы Липшица.
        """
        if self.recalcM is not True:
            return
        for item in self.searchData:
            self.CalculateM(item, item.GetLeft())
        self.recalcM = False

    def RecalcAllCharacteristics(self) -> None:
        r"""
        Пересчёт характеристик для всех поисковых интервалов.
        """
        if self.recalcR is not True:
            return
        self.searchData.ClearQueue()
        for item in self.searchData:  # Должно работать...
            self.CalculateGlobalR(item, item.GetLeft())
            # self.CalculateLocalR(item)
        self.searchData.RefillQueue()
        self.recalcR = False

    def CalculateNextPointCoordinate(self, point: SearchDataItem) -> float:
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}` в заданном интервале :math:`[x_{t-1},x_t]`.

        :param point: интервал, заданный его правой точкой :math:`x_t`.

        :return: точка нового испытания :math:`x^{k+1}` в этом интервале.
        """
        # https://github.com/MADZEROPIE/ags_nlp_solver/blob/cedcbcc77aa08ef1ba591fc7400c3d558f65a693/solver/src/solver.cpp#L420
        left = point.GetLeft()
        if left is None:
            print("CalculateNextPointCoordinate: Left point is NONE")
            raise Exception("CalculateNextPointCoordinate: Left point is NONE")
        xl = left.GetX()
        xr = point.GetX()
        idl = left.GetIndex()
        idr = point.GetIndex()
        if idl == idr and idl >= 0:
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
        if self.recalcM is True:
            self.RecalcM()
        if self.recalcR is True:
            self.RecalcAllCharacteristics()

        old = self.searchData.GetDataItemWithMaxGlobalR()
        self.min_delta = min(old.delta, self.min_delta)
        newx = self.CalculateNextPointCoordinate(old)
        newy = self.evolvent.GetImage(newx)
        new = copy.deepcopy(SearchDataItem(Point(newy, []), newx,
                                           functionValues=[FunctionValue()] * self.numberOfAllFunctions))

        # Обновление числа испытаний
        self.searchData.solution.numberOfGlobalTrials += 1

        return new, old

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        try:
            point = self.task.Calculate(point, 0)
            point.SetZ(point.functionValues[0].value)
            point.SetIndex(0)
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
        if curr_point is None:
            print("CalculateM: curr_point is None")
            raise RuntimeError("CalculateM: curr_point is None")
        if left_point is None:
            return
        index = curr_point.GetIndex()
        if left_point.GetIndex() == index and index >= 0:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left_point.GetZ() - curr_point.GetZ()) / curr_point.delta
            if m > self.M[index]:
                self.M[index] = m
                self.recalcR = True

    # def CalculateM(self, point: SearchDataItem):  # В python нет такой перегрузки функций, надо менять название
    #     self.CalculateM(point, point.GetLeft())

    def CalculateGlobalR(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление глобальной характеристики интервала [left_point, curr_point].

        :param curr_point: правая точка интервала.
        :param left_point: левая точка интервала.
        """
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

    def RenewSearchData(self, newpoint: SearchDataItem, oldpoint: SearchDataItem) -> None:
        """
        Метод обновляет всю поисковую информацию: длины интервалов, константы Гёльдера, все характеристики и вставляет
          новую точку в хранилище.

        :param newpoint: новая точка
        :param oldpoint: правая точка интервала, которому принадлежит новая точка
        """

        # oldpoint.delta = Method.CalculateDelta(newpoint.GetX(), oldpoint.GetX(), self.dimension)
        # newpoint.delta = Method.CalculateDelta(oldpoint.GetLeft().GetX(), newpoint.GetX(), self.dimension)

        oldpoint.delta = self.CalculateDelta(newpoint, oldpoint, self.dimension)
        newpoint.delta = self.CalculateDelta(oldpoint.GetLeft(), newpoint, self.dimension)

        self.CalculateM(newpoint, oldpoint.GetLeft())
        self.CalculateM(oldpoint, newpoint)

        self.CalculateGlobalR(newpoint, oldpoint.GetLeft())
        self.CalculateGlobalR(oldpoint, newpoint)

        self.searchData.InsertDataItem(newpoint, oldpoint)

    def UpdateOptimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.

        :param point: точка нового испытания.
        """
        if self.best is None or self.best.GetIndex() < point.GetIndex():
            self.best = point
            self.recalcR = True
            self.Z[point.GetIndex()] = point.GetZ()
        elif self.best.GetIndex() == point.GetIndex() and point.GetZ() < self.best.GetZ():
            self.best = point
            self.recalcR = True
            self.Z[point.GetIndex()] = point.GetZ()
        self.searchData.solution.bestTrials[0] = self.best

    def FinalizeIteration(self) -> None:
        r"""
        Заканчивает итерацию, обновляет счётчик итераций.
        """
        self.iterationsCount += 1

    def GetIterationsCount(self) -> int:
        r"""
        Возвращает число выполненных итераций.

        :return:  число выполненных итераций.
        """
        return self.iterationsCount

    def GetOptimumEstimation(self) -> SearchDataItem:
        r"""
        Возвращает оценку оптимума.

        :return: текущая оценка оптимума.
        """
        return self.best
