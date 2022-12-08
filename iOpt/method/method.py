from __future__ import annotations

from typing import Tuple

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.trial import Point, Trial
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.method.optim_task import OptimizationTask
from iOpt.solver_parametrs import SolverParameters

import copy


class Method:
    """
    Класс Method содержит методы алгоритма глобального поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        r"""
        Конструктор класса Method

        :param parameters: Параметры поиска оптимальных решений
        :param task: Обёртка решаемой задачи.
        :param evolvent: Развёртка.
        :param searchData: Хранилище данных.
        """
        self.stop: bool = False
        self.recalc: bool = True
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

    @property
    def min_delta(self):
        return self.searchData.solution.solutionAccuracy

    @min_delta.setter
    def min_delta(self, val):
        self.searchData.solution.solutionAccuracy = val

    @staticmethod
    def CalculateDelta(lx: float, rx: float, dimension: int) -> float:
        """
        Возвращает расстояние между двумя точками развертки.
        :params lx: Левая точка
        :params rx: Правая точка
        :params dimension: Размерность пространства
        """
        return pow(rx - lx, 1.0 / dimension)

    def FirstIteration(self) -> None:
        r"""
        Метод, производящий первую итерацию АГП.
        """
        self.iterationsCount = 1
        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        x: float = 0.5
        y = Point(self.evolvent.GetImage(x), None)
        middle = SearchDataItem(y, x)
        left = SearchDataItem(Point(self.evolvent.GetImage(0.0), None), 0.0)
        right = SearchDataItem(Point(self.evolvent.GetImage(1.0), None), 1.0)

        left.delta = 0
        middle.delta = Method.CalculateDelta(left.GetX(), middle.GetX(), self.dimension)
        right.delta = Method.CalculateDelta(middle.GetX(), right.GetX(), self.dimension)

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

    def CheckStopCondition(self) -> bool:
        r"""
        Проверка условия остановки.
        Алгоритм должен завершить работу, когда достигнута точность eps или превышен лимит итераций.
        """
        if self.min_delta < self.parameters.eps or self.iterationsCount >= self.parameters.itersLimit:
            self.stop = True
        else:
            self.stop = False

        return self.stop

    def RecalcAllCharacteristics(self) -> None:
        r"""
        Пересчёт характеристик для всех испытаний, когда поднят флаг recalc.
        """
        if self.recalc is not True:
            return
        self.searchData.ClearQueue()
        for item in self.searchData:  # Должно работать...
            self.CalculateGlobalR(item, item.GetLeft())
            # self.CalculateLocalR(item)
        self.searchData.RefillQueue()
        self.recalc = False

    def CalculateNextPointCoordinate(self, point: SearchDataItem) -> float:
        r"""
        Подсчёт координаты новой точки по правилу АГП для выбранного интервала
        :param point: Выбранный интервал
        :return: Координата точки в этом интервале.
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
        Подсчёт новой точки испытания.
        :return: Новая точка и покрывающий интервал.
        """
        if self.recalc is True:
            self.RecalcAllCharacteristics()

        old = self.searchData.GetDataItemWithMaxGlobalR()
        self.min_delta = min(old.delta, self.min_delta)
        newx = self.CalculateNextPointCoordinate(old)
        newy = self.evolvent.GetImage(newx)
        new = copy.deepcopy(SearchDataItem(Point(newy, []), newx))
        return new, old

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Подсчёт значений функций в данной точке.
        :params point: Точка, в которой надо посчитать значение функционала.
        :return: Точка с посчитанным функционалом.
        """
        # point.functionValues = np.array(shape=self.task.problem.numberOfObjectives, dtype=FunctionValue)
        # for func_id in range(self.task.problem.numberOfObjectives):  # make Calculate Objectives?
        #    self.task.Calculate(point, func_id)  # SetZ, BUT

        # Завернуть в цикл для индексной схемы
        point = self.task.Calculate(point, 0)
        point.SetZ(point.functionValues[0].value)
        point.SetIndex(0)

        # Обновление числа испытаний
        self.searchData.solution.numberOfGlobalTrials += 1
        return point

    def CalculateM(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Считает константу Хёльдера между curr_point и left_point.
        :params curr_point: Правая точка интервала
        :params left_point: Левая точка интервала
        """
        if curr_point is None:
            print("CalculateM: curr_point is None")
            raise RuntimeError("CalculateM: curr_point is None")
        if left_point is None:
            return
        index = curr_point.GetIndex()
        if left_point.GetIndex() == index:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left_point.GetZ() - curr_point.GetZ()) / curr_point.delta
            if m > self.M[index]:
                self.M[index] = m
                self.recalc = True

    # def CalculateM(self, point: SearchDataItem):  # В python нет такой перегрузки функций, надо менять название
    #     self.CalculateM(point, point.GetLeft())

    def CalculateGlobalR(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Считает глобальную характеристику для curr_point в предположении, что левая точка интервала - left_point.

        :params curr_point: Правая точка интервала
        :params left_point: Левая точка интервала
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
        if left_point.GetIndex() == curr_point.GetIndex():
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
        Обновляет длину интервалов, константу Гёльдера, все характеристики и вставляет новую точку в хранилище.

        :params newpoint: Новая точка
        :params oldpoint: Её покрывающий интервал
        """

        oldpoint.delta = Method.CalculateDelta(newpoint.GetX(), oldpoint.GetX(), self.dimension)
        newpoint.delta = Method.CalculateDelta(oldpoint.GetLeft().GetX(), newpoint.GetX(), self.dimension)

        self.CalculateM(newpoint, oldpoint.GetLeft())
        self.CalculateM(oldpoint, newpoint)

        self.CalculateGlobalR(newpoint, oldpoint.GetLeft())
        self.CalculateGlobalR(oldpoint, newpoint)

        self.searchData.InsertDataItem(newpoint, oldpoint)

    def UpdateOptimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.
        :params point: Новая точка
        """
        if self.best is None or self.best.GetIndex() < point.GetIndex():
            self.best = point
            self.recalc = True
            self.Z[point.GetIndex()] = point.GetZ()
        elif self.best.GetIndex() == point.GetIndex() and point.GetZ() < self.best.GetZ():
            self.best = point
            self.recalc = True
            self.Z[point.GetIndex()] = point.GetZ()
        self.searchData.solution.bestTrials[0] = self.best

    def FinalizeIteration(self) -> None:
        r"""
        Заканчивает итерацию, обновляет счётчик итераций.
        """
        self.iterationsCount += 1

    def GetIterationsCount(self) -> int:
        r"""
        Возвращает количество выполненных итераций.
        :return:  self.iterationsCount
        """
        return self.iterationsCount

    def GetOptimumEstimation(self) -> SearchDataItem:
        r"""
        Возвращает оценку оптимума.
        :return: self.best
        """
        return self.best
