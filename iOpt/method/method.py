from __future__ import annotations

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.trial import Point
from search_data import SearchData
from search_data import SearchDataItem
from optim_task import OptimizationTask
from iOpt.solver import SolverParameters


# TODO: Привести комментарии в порядок

class Method:
    stop: bool = False
    recalc: bool = True
    iterationsCount: int = 0
    best: SearchDataItem = None
    min_delta: np.double = np.infty

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
        middle.delta = 0.5  # / self.dimension ???
        right.delta = 0.5  # / self.dimension ???

        # Вычисление значения функции в 0.5
        self.CalculateFunctionals(middle)
        self.best = middle

        # TODO: заменить на InsertFirstItem(s) и убрать SetLeft/SetRight
        # вставить left  и right, потом middle
        middle.SetLeft(left)
        middle.SetRight(right)
        left.SetRight(middle)
        right.SetLeft(middle)
        self.searchData.InsertDataItem(right)
        self.searchData.InsertDataItem(middle, right)
        self.searchData.InsertDataItem(left, middle)

        # Вычисление характеристик
        recalc = True
        self.RecalcAllCharacteristics()

    def CheckStopCondition(self):
        self.stop = self.min_delta < self.parameters.eps
        return self.stop

    def RecalcAllCharacteristics(self):
        if self.recalc is not True:
            return
        self.searchData.ClearQueue()
        for item in self.searchData:  # Должно работать...
            self.CalculateGlobalR(item)
            # self.CalculateLocalR(item)
        self.searchData.RefillQueue()
        self.recalc = False

    def CalculateNextPointCoordinate(self, point: SearchDataItem):
        # https://github.com/MADZEROPIE/ags_nlp_solver/blob/cedcbcc77aa08ef1ba591fc7400c3d558f65a693/solver/src/solver.cpp#L420
        x = 0
        left = point.GetLeft()
        idl = left.GetIndex()
        idr = point.GetIndex()
        if idl == idr:
            v = idr
            dif = left.GetZ() - point.GetZ()
            dg = -1.0
            if dif > 0:
                dg = 1.0

            x = 0.5 * (left.GetX() + point.GetX())
            x -= 0.5 * dg * pow(abs(dif) / self.M[v], self.task.problem.numberOfFloatVariables) / self.parameters.r

        else:
            x = 0.5 * (point.GetX() + left.GetX())
        return x

    def CalculateIterationPoints(self) -> (SearchDataItem, SearchDataItem):  # return  (new, old)
        if self.recalc is True:
            self.RecalcAllCharacteristics()
        old = self.searchData.GetDataItemWithMaxGlobalR()
        newx = self.CalculateNextPointCoordinate(old)
        newy = self.evolvent.GetImage(newx)
        new = SearchDataItem(newy, newx)
        return (new, old)

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        # point.functionValues = np.array(shape=self.task.problem.numberOfObjectives, dtype=FunctionValue)
        # for func_id in range(self.task.problem.numberOfObjectives):  # make Calculate Objectives?
        #    point.functionValues[func_id] = self.task.Calculate(point, func_id)  # SetZ, BUT

        # Завернуть в цикл для индексной схемы
        point = self.task.Calculate(point, 0)
        point.SetZ(point.functionValues[0])
        point.SetIndex(0)
        return point

    def CalculateM(self, point: SearchDataItem):
        left = point.GetLeft()
        if left is None:
            return
        index = point.GetIndex()
        if left.GetIndex() == index:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left.GetZ() - point.GetZ()) / point.delta
            if m > self.M[index]:
                self.M[index] = m
                self.recalc = True

    def CalculateGlobalR(self, point: SearchDataItem):
        if point.GetIndex() < 0:
            point.globalR = -np.infty
            return None
        left = point.GetLeft()
        zl = left.GetZ()
        zr = point.GetZ()
        r = self.parameters.r
        deltax = point.delta
        if left.GetIndex() == point.GetIndex():
            v = point.GetIndex()
            globalR = deltax + (zr - zl) * (zr - zl) / (deltax * self.M[v] * self.M[v] * r * r) - \
                      2 * (zr + zl - 2 * self.Z[v]) / (r * self.M[v])
        elif left.GetIndex() < point.GetIndex():
            v = point.GetIndex()
            globalR = 2 * deltax - 4 * (zr - self.Z[v]) / (r * self.M[v])
        else:
            v = left.GetIndex()
            globalR = 2 * deltax - 4 * (zl - self.Z[v]) / (r * self.M[v])
        point.globalR = globalR

    def RenewSearchData(self, point: (SearchDataItem, SearchDataItem)):
        # calc M, R(?), insert, Z.

        self.CalculateM(point[0])

        self.CalculateGlobalR(point[0])
        # point[0].SetLeft(point[1].GetLeft())
        # point[1].SetRight(point[1])
        # point[1].SetLeft(point[0])
        point[1].delta = pow(point[1].GetX() - point[0].GetX(), 1.0 / self.dimension)
        point[0].delta = pow(point[0].GetX() - point[0].GetLeft().GetX(), 1.0 / self.dimension)
        self.min_delta = min(point[0].delta, self.min_delta)
        self.min_delta = min(point[1].delta, self.min_delta)
        self.searchData.InsertDataItem(point[0], point[1])

    def UpdateOptimum(self, point: SearchDataItem):
        if self.best.GetIndex() < point.GetIndex():  # CHECK INDEX
            self.best = point
            self.recalc = True
        elif self.best.GetIndex() == point.GetIndex() and point.GetZ() < self.best.GetZ():
            self.best = point
            self.recalc = True

    def FinalizeIteration(self):
        self.iterationsCount += 1

    def GetIterationsCount(self):
        return self.iterationsCount

    def GetOptimumEstimation(self):
        return self.best
