from __future__ import annotations

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.trial import Point, Trial
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.method.optim_task import OptimizationTask
from iOpt.solver_parametrs import SolverParameters


# TODO: Привести комментарии в порядок

class Method:
    stop: bool = False
    recalc: bool = True
    iterationsCount: int = 0
    best: SearchDataItem = None

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                 ):
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
        # change to np.array, but indexing np is slower
        self.M = [1.0 for _ in range(task.problem.numberOfObjectives + task.problem.numberOfConstraints)]
        self.Z = [np.infty for _ in range(task.problem.numberOfObjectives + task.problem.numberOfConstraints)]
        self.dimension = task.problem.numberOfFloatVariables  # А ДЛЯ ДИСКРЕТНЫХ?
        # self.best: Trial = SearchData.solution.bestTrials[0]  # Это ведь ССЫЛКА, ДА?
        self.searchData.solution.solutionAccuracy = np.infty

    @property
    def min_delta(self):
        return self.searchData.solution.solutionAccuracy

    @min_delta.setter
    def min_delta(self, val):
        self.searchData.solution.solutionAccuracy = val

    def FirstIteration(self):
        self.iterationsCount = 1
        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        x: np.double = 0.5
        y = Point(self.evolvent.GetImage(x), None)
        middle = SearchDataItem(y, x)
        left = SearchDataItem(Point(self.evolvent.GetImage(0.0), None), 0.0)
        right = SearchDataItem(Point(self.evolvent.GetImage(1.0), None), 1.0)

        left.delta = 0
        middle.delta = 0.5  # / self.dimension  # ???
        right.delta = 0.5  # / self.dimension  # ???

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

    def CheckStopCondition(self):
        self.stop = self.min_delta < self.parameters.eps
        return self.stop

    def RecalcAllCharacteristics(self):
        if self.recalc is not True:
            return
        self.searchData.ClearQueue()
        for item in self.searchData:  # Должно работать...
            self.CalculateGlobalR(item, item.GetLeft())
            # self.CalculateLocalR(item)
        self.searchData.RefillQueue()
        self.recalc = False

    def CalculateNextPointCoordinate(self, point: SearchDataItem):
        # https://github.com/MADZEROPIE/ags_nlp_solver/blob/cedcbcc77aa08ef1ba591fc7400c3d558f65a693/solver/src/solver.cpp#L420
        left = point.GetLeft()
        if left is None:
            raise "CalculateNextPointCoordinate: Left point is NONE"
        xl = left.GetX()
        xr = point.GetX()
        idl = left.GetIndex()
        idr = point.GetIndex()
        if idl == idr:
            v = idr
            dif = left.GetZ() - point.GetZ()
            dg = -1.0
            if dif > 0:
                dg = 1.0

            x = 0.5 * (xl + xr)
            x -= 0.5 * dg * pow(abs(dif) / self.M[v], self.task.problem.numberOfFloatVariables) / self.parameters.r

        else:
            x = 0.5 * (xl + xr)
        if x <= xl or x >= xr:
            raise "CalculateNextPointCoordinate: x is outside of interval"
        return x

    def CalculateIterationPoint(self) -> (SearchDataItem, SearchDataItem):  # return  (new, old)
        """
        Calculate new iteration point.
        :return: a pair (tuple) of SearchDataItem. pair[0] -> new point (left), pair[1] -> old point (right)
        """
        if self.recalc is True:
            self.RecalcAllCharacteristics()

        old = self.searchData.GetDataItemWithMaxGlobalR()
        self.min_delta = min(old.delta, self.min_delta)
        newx = self.CalculateNextPointCoordinate(old)
        newy = self.evolvent.GetImage(newx)
        new = SearchDataItem(Point(newy, []), newx)
        return (new, old)

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
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

    def CalculateM(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        """
        Calculate holder constant of curr_point in assumption that curr_point.left should be left_point
        """
        if curr_point is None:
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

    def CalculateGlobalR(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        """
        Calculate Global characteristic of curr_point in assumption that curr_point.left should be left_point
        """
        if curr_point is None:
            raise "CalculateGlobalR: Curr point is NONE"
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

    # def CalculateGlobalR(self, curr_point: SearchDataItem):
    #     self.CalculateGlobalR(curr_point, curr_point.GetLeft())

    def RenewSearchData(self, newpoint: SearchDataItem, oldpoint: SearchDataItem):
        """
        :params: pair of SearchDataItem. newpoint -> new point (left), oldpoint -> old point (right)
        Update delta, M, R and insert points to searchData.
        """

        oldpoint.delta = pow(oldpoint.GetX() - newpoint.GetX(), 1.0 / self.dimension)
        newpoint.delta = pow(newpoint.GetX() - oldpoint.GetLeft().GetX(), 1.0 / self.dimension)

        self.CalculateM(newpoint, oldpoint.GetLeft())
        self.CalculateM(oldpoint, newpoint)

        self.CalculateGlobalR(newpoint, oldpoint.GetLeft())
        self.CalculateGlobalR(oldpoint, newpoint)

        self.searchData.InsertDataItem(newpoint, oldpoint)

    def UpdateOptimum(self, point: SearchDataItem):
        if self.best is None or self.best.GetIndex() < point.GetIndex():  # CHECK INDEX
            self.best = point
            self.recalc = True
            self.Z[point.GetIndex()] = point.GetZ()
        elif self.best.GetIndex() == point.GetIndex() and point.GetZ() < self.best.GetZ():
            self.best = point
            self.recalc = True
            self.Z[point.GetIndex()] = point.GetZ()

    def FinalizeIteration(self):
        self.iterationsCount += 1

    def GetIterationsCount(self):
        return self.iterationsCount

    def GetOptimumEstimation(self):
        return self.best
