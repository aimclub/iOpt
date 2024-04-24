from __future__ import annotations

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataItem
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.method import Method


class GridSearchMethod(Method):
    """
    The Method class contains an implementation of the Grid Search Algorithm
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData
                 ):
        super(GridSearchMethod, self).__init__(parameters, task, evolvent, search_data)

        self.current_pont_index = [0 for _ in range(task.problem.number_of_float_variables)]
        self.count_step_in_float_variables: list[int] = []
        self.h_discrete_variables: list[int] = []
        self.y_values: list[list[float]] = [[]]
        self.nead_iteration_count = 1

        for i in range(self.task.problem.number_of_float_variables):
            self.count_step_in_float_variables.append(int((self.task.problem.upper_bound_of_float_variables[i] -
                                                           self.task.problem.lower_bound_of_float_variables[
                                                               i]) / self.parameters.eps))
            self.y_values.append([])
            for j in range(self.count_step_in_float_variables[i]):
                self.y_values[i].append(self.parameters.eps * j + self.task.problem.lower_bound_of_float_variables[i])
            self.current_pont_index[i] = 0
            self.nead_iteration_count *= self.count_step_in_float_variables[i]

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

        x = 0.5 * (xl + xr)

        if x <= xl or x >= xr:
            print(f"CalculateNextPointCoordinate: x is outside of interval {x} {xl} {xr}")
            raise Exception("CalculateNextPointCoordinate: x is outside of interval")
        return x

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

        deltax = curr_point.delta

        global_r = deltax

        curr_point.globalR = global_r
