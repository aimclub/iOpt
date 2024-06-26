from __future__ import annotations

import copy
import math
import sys
from typing import Tuple
from time import time

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.default_calculator import DefaultCalculator
from iOpt.method.index_method_evaluate import IndexMethodEvaluate
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue


class Method:
    """
    The Method class contains an implementation of the Global Search Algorithm
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator = None
                 ):
        r"""
        Method class constructor

        :param parameters: parameters for solving the optimization problem.
        :param task: problem wrapper.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: data structure for storing accumulated search information.
        :param calculator: class containing trial methods (parallel and/or inductive circuit)
        """

        self.stop: bool = False
        self.recalcR: bool = True
        self.recalcM: bool = True
        self.iterations_count: int = 0
        self.best: SearchDataItem = None

        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.search_data = search_data

        self.M = [1.0 for _ in range(1 + task.problem.number_of_constraints)]
        self.Z = [np.infty for _ in range(1 + task.problem.number_of_constraints)]
        self.dimension = task.problem.number_of_float_variables
        self.search_data.solution.solution_accuracy = np.infty
        self.numberOfAllFunctions = task.problem.number_of_objectives + task.problem.number_of_constraints

        if calculator is None:
            self.calculator = DefaultCalculator(IndexMethodEvaluate(self.task), parameters=self.parameters)
        else:
            self.calculator = calculator

    @property
    def min_delta(self):
        return self.search_data.solution.solution_accuracy

    @min_delta.setter
    def min_delta(self, val):
        self.search_data.solution.solution_accuracy = val

    def calculate_delta(self, l_point: SearchDataItem, r_point: SearchDataItem, dimension: int) -> float:
        """
        Compute the Gelder distance in the Gelder metric between two points on the segment [0,1],
          obtained by dimensionality reduction

        :param l_point: left point.
        :param r_point: right point.
        :param dimension: dimensionality of the original space.

        :return: helder distance between lx and rx.
        """
        return pow(r_point.get_x() - l_point.get_x(), 1.0 / dimension)

    def first_iteration(self) -> list[SearchDataItem]:
        r"""
        Perform the first iteration of the Global Search Algorithm
        """

        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        left = SearchDataItem(Point(self.evolvent.get_image(0.0), []), 0.,
                              function_values=[FunctionValue()] * self.numberOfAllFunctions)
        right = SearchDataItem(Point(self.evolvent.get_image(1.0), []), 1.0,
                               function_values=[FunctionValue()] * self.numberOfAllFunctions)

        items: list[SearchDataItem] = []

        if self.parameters.start_point:
            number_of_point: int = self.parameters.number_of_parallel_points - 1
            h: float = 1.0 / (number_of_point + 1)

            ystart_point = Point(copy.copy(self.parameters.start_point.float_variables), [])
            xstart_point = self.evolvent.get_inverse_image(self.parameters.start_point.float_variables)

            itemstart_point = SearchDataItem(ystart_point, xstart_point,
                                             function_values=[FunctionValue()] * self.numberOfAllFunctions)

            is_add_start_point: bool = False

            for i in range(number_of_point):
                x = h * (i + 1)
                y = Point(self.evolvent.get_image(x), [])
                item = SearchDataItem(y, x,
                                      function_values=[FunctionValue()] * self.numberOfAllFunctions)
                if x < xstart_point < h * (i + 2):
                    items.append(item)
                    items.append(itemstart_point)
                    is_add_start_point = True
                else:
                    items.append(item)

            if not is_add_start_point:
                items.append(itemstart_point)
        else:

            number_of_point: int = self.parameters.number_of_parallel_points
            h: float = 1.0 / (number_of_point + 1)

            for i in range(number_of_point):
                x = h * (i + 1)
                y = Point(self.evolvent.get_image(x), [])
                item = SearchDataItem(y, x,
                                      function_values=[FunctionValue()] * self.numberOfAllFunctions)
                items.append(item)

        self.calculator.calculate_functionals_for_items(items)

        for item in items:
            self.update_optimum(item)

        left.delta = 0
        self.calculate_global_r(left, None)

        items[0].delta = self.calculate_delta(left, items[0], self.dimension)
        self.calculate_global_r(items[0], left)
        for id_item, item in enumerate(items):
            if id_item > 0:
                items[id_item].delta = self.calculate_delta(items[id_item - 1], items[id_item], self.dimension)
                self.calculate_global_r(items[id_item], items[id_item - 1])
                self.calculate_m(items[id_item], items[id_item - 1])

        right.delta = self.calculate_delta(items[-1], right, self.dimension)
        self.calculate_global_r(right, items[-1])

        # вставить left  и right, потом middle
        self.search_data.insert_first_data_item(left, right)

        for item in items:
            self.search_data.insert_data_item(item, right)

        self.recalcR = True
        self.recalcM = True

        self.iterations_count = len(items)
        self.search_data.solution.number_of_global_trials = len(items)

        return items

    def check_stop_condition(self) -> bool:
        r"""
        Check the stop condition.
        The algorithm should terminate when eps accuracy is reached or the iteration limit is exceeded

        :return: True if the stop criterion is met; False otherwise.
        """
        if self.min_delta < self.parameters.eps or self.iterations_count >= self.parameters.global_method_iteration_count:
            self.stop = True
        else:
            self.stop = False

        return self.stop

    def recalc_m(self) -> None:
        r"""
        Recalculate the estimate of the Lipschitz constant
        """
        if self.recalcM is not True:
            return
        for item in self.search_data:
            self.calculate_m(item, item.get_left())
        self.recalcM = False

    def recalc_all_characteristics(self) -> None:
        r"""
        Recalculate of features for all search intervals
        """
        if self.recalcR is not True:
            return
        self.search_data.clear_queue()
        for item in self.search_data:  # Должно работать...
            self.calculate_global_r(item, item.get_left())
        self.search_data.refill_queue()
        self.recalcR = False

    def calculate_next_point_coordinate(self, point: SearchDataItem) -> float:
        r"""
        Compute the point of a new trial :math:`x^{k+1}` in a given interval :math:`[x_{t-1},x_t]`

        :param point: interval given by its right point :math:`x_t`.

        :return: the point of a new trial :math:`x^{k+1}` in this interval.
        """
        # https://github.com/MADZEROPIE/ags_nlp_solver/blob/cedcbcc77aa08ef1ba591fc7400c3d558f65a693/solver/src/solver.cpp#L420
        left = point.get_left()
        if left is None:
            print("CalculateNextPointCoordinate: Left point is NONE")
            raise Exception("CalculateNextPointCoordinate: Left point is NONE")
        xl = left.get_x()
        xr = point.get_x()
        idl = left.get_index()
        idr = point.get_index()
        if idl == idr and idl >= 0:
            v = idr
            dif = point.get_z() - left.get_z()
            dg = -1.0
            if dif > 0:
                dg = 1.0

            x = 0.5 * (xl + xr)
            x -= 0.5 * dg * pow(abs(dif) / self.M[v], self.task.problem.number_of_float_variables) / self.parameters.r

        else:
            x = 0.5 * (xl + xr)
        if x <= xl or x >= xr:
            print(f"CalculateNextPointCoordinate: x is outside of interval {x} {xl} {xr}")
            raise Exception("CalculateNextPointCoordinate: x is outside of interval")
        return x

    def calculate_iteration_point(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        r"""
        Calculate the point of a new trial :math:`x^{k+1}`

        :return: :math:`x^{k+1}` - new trial point, и :math:`x_t` - left interval point :math:`[x_{t-1},x_t]`,
          to which belongs :math:`x^{k+1}`, that is :math:`x^{k+1} \in [x_{t-1},x_t]`.
        """
        if self.recalcM is True:
            self.recalc_m()
        if self.recalcR is True:
            self.recalc_all_characteristics()

        old = self.search_data.get_data_item_with_max_global_r()
        self.min_delta = min(old.delta, self.min_delta)
        newx = self.calculate_next_point_coordinate(old)
        newy = self.evolvent.get_image(newx)
        new = copy.deepcopy(SearchDataItem(Point(newy, []), newx,
                                           function_values=[FunctionValue()] * self.numberOfAllFunctions))

        # Обновление числа испытаний
        self.search_data.solution.number_of_global_trials += 1

        return new, old

    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Perform a search trial at a given point

        :param point: the point at which the trial is to be performed.

        :return: the point at which the trial results are saved.
        """
        try:
            self.calculator.calculate_functionals_for_items([point])
        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-10)

        return point

    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Calculate an estimate of the Gelder constant between curr_point and left_point

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """
        if curr_point is None:
            print("CalculateM: curr_point is None")
            raise RuntimeError("CalculateM: curr_point is None")
        if left_point is None:
            return
        index = curr_point.get_index()
        if left_point.get_index() == index and index >= 0:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left_point.get_z() - curr_point.get_z()) / curr_point.delta
            if m > self.M[index]:
                self.M[index] = m
                self.recalcR = True

    def calculate_global_r(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Calculate the global characteristic of an interval [left_point, curr_point]

        :param curr_point: right interval point.
        :param left_point: left interval point.
        """
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

    def renew_search_data(self, newpoint: SearchDataItem, oldpoint: SearchDataItem) -> None:
        """
        Update all search information: interval lengths, Gölder constants, all characteristics and inserts
a new point into the repository

        :param newpoint: new point.
        :param oldpoint: right point of the interval to which the new point belongs.
        """

        oldpoint.delta = self.calculate_delta(newpoint, oldpoint, self.dimension)
        newpoint.delta = self.calculate_delta(oldpoint.get_left(), newpoint, self.dimension)

        self.calculate_m(newpoint, oldpoint.get_left())
        self.calculate_m(oldpoint, newpoint)

        self.calculate_global_r(newpoint, oldpoint.get_left())
        self.calculate_global_r(oldpoint, newpoint)

        self.search_data.insert_data_item(newpoint, oldpoint)

    def update_optimum(self, point: SearchDataItem) -> None:
        r"""
        Update the optimum estimate

        :param point: point of a new trial.
        """
        if self.best is None or self.best.get_index() < point.get_index():
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()
        elif self.best.get_index() == point.get_index() and point.get_z() < self.best.get_z():
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()
        self.search_data.solution.best_trials[0] = self.best

    def finalize_iteration(self) -> None:
        r"""
        End the iteration, updates the iteration counter
        """
        self.search_data.get_last_item().creation_time = time()
        self.search_data.get_last_item().iterationNumber = self.iterations_count  # будет ли работать в параллельном случае?
        self.iterations_count += 1

    def get_iterations_count(self) -> int:
        r"""
        Return the number of iterations performed

        :return:  number of iterations performed.
        """
        return self.iterations_count

    def get_optimum_estimation(self) -> SearchDataItem:
        r"""
        Return an estimate of the optimum

        :return: current estimate of the optimum.
        """
        return self.best
