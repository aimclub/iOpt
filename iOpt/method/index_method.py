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
    The Method class contains an implementation of the Global Search Algorithm
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData
                 ):
        super(IndexMethod, self).__init__(parameters, task, evolvent, search_data)

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Perform a search trial at a given point

        :param point: the point at which the trial is to be performed.

        :return: the point at which the trial results are saved.
        """
        try:
            number_of_constraints = self.task.problem.number_of_constraints
            for i in range(number_of_constraints):
                point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)  # ???
                point = self.task.calculate(point, i)
                point.set_z(point.function_values[i].value)
                point.set_index(i)
                if point.get_z() > 0:
                    return point
            point.function_values[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV, number_of_constraints)
            point = self.task.calculate(point, number_of_constraints)
            point.set_z(point.function_values[number_of_constraints].value)
            point.set_index(number_of_constraints)
        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-10)

        return point

    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Compute an estimate of the Gelder constant between curr_point and left_point

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """
        # Обратить внимание на вычисление расстояния, должен использоваться метод CalculateDelta
        if curr_point is None:
            print("CalculateM: curr_point is None")
            raise RuntimeError("CalculateM: curr_point is None")
        if left_point is None:
            return
        index = curr_point.get_index()
        if index < 0:
            return
        m = 0.0
        if left_point.get_index() == index:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left_point.get_z() - curr_point.get_z()) / curr_point.delta
        else:
            # Ищем слева
            other_point = left_point
            while (other_point is not None) and (other_point.get_index() < curr_point.get_index()):
                other_point = other_point.get_left()
            if other_point is not None and other_point.get_index() >= 0:
                # print(index)
                m = abs(other_point.function_values[index].value - curr_point.get_z()) / \
                    self.calculate_delta(other_point, curr_point, self.dimension)
            # Ищем справа
            other_point = left_point.get_right()
            if other_point is not None and other_point is curr_point:  # возможно только при пересчёте M
                other_point = other_point.get_right()
            while (other_point is not None) and (other_point.get_index() < curr_point.get_index()):
                other_point = other_point.get_right()
            if other_point is not None and other_point.get_index() >= 0:
                m = max(m, abs(curr_point.get_z() - other_point.function_values[index].value) / \
                        self.calculate_delta(curr_point, other_point, self.dimension))

        if m > self.M[index] or (self.M[index] == 1.0 and m > 1e-12):
            self.M[index] = m
            self.recalcR = True

    def calculate_global_r(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Calculate the global characteristic of an interval [left_point, curr_point]

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """

        # Сюда переедет целиком calculate_global_r из Method, а там останется только случай с равными индексами
        if curr_point is None:
            print("calculate_global_r: Curr point is NONE")
            raise Exception("calculate_global_r: Curr point is NONE")
        if left_point is None:
            curr_point.globalR = -np.infty
            return None
        zl = left_point.get_z()
        zr = curr_point.get_z()
        r = self.parameters.r
        deltax = curr_point.delta

        if left_point.get_index() < 0 and curr_point.get_index() < 0:
            global_r = 2 * deltax - 4 * math.fabs(self.Z[0]) / (r * self.M[0])
        elif left_point.get_index() == curr_point.get_index():
            v = curr_point.get_index()
            global_r = deltax + (zr - zl) * (zr - zl) / (deltax * self.M[v] * self.M[v] * r * r) - \
                       2 * (zr + zl - 2 * self.Z[v]) / (r * self.M[v])
        elif left_point.get_index() < curr_point.get_index():
            v = curr_point.get_index()
            global_r = 2 * deltax - 4 * (zr - self.Z[v]) / (r * self.M[v])
        else:
            v = left_point.get_index()
            global_r = 2 * deltax - 4 * (zl - self.Z[v]) / (r * self.M[v])
        curr_point.globalR = global_r

    def update_z(self, point: SearchDataItem) -> None:
        for i in range(point.get_index()):
            if self.Z[i] > point.function_values[i].value:
                self.Z[i] = point.function_values[i].value
                self.recalcR = True

    def recalc_all_characteristics(self) -> None:
        for i in range(self.best.get_index()):
            self.Z[i] = -self.M[i] * self.parameters.eps_r
        self.Z[self.best.get_index()] = self.best.get_z()
        super().recalc_all_characteristics()

    def update_optimum(self, point: SearchDataItem) -> None:
        r"""
        Update the estimate of the optimum

        :param point: the point of a new trial.
        """

        if self.best is None or self.best.get_index() < point.get_index() or (
                self.best.get_index() == point.get_index() and point.get_z() < self.best.get_z()):
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()
        # self.UpdateZ(point)
        self.search_data.solution.best_trials[0] = self.best
