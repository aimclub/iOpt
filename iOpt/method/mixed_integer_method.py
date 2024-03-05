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
    The Method class contains an implementation of the Global Search Algorithm
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator = None
                 ):
        super(MixedIntegerMethod, self).__init__(parameters, task, evolvent, search_data, calculator)

        # u = {i, j, k}, i = {0, 1, 2}, j = {0, 1}, k = {0, 1, 2, 3, 4} -> 3*2*4=24

        list_discrete_values = list(task.problem.discrete_variable_values)
        self.discreteParameters = list(itertools.product(*list_discrete_values))
        # определяем количество сочетаний параметров
        self.numberOfParameterCombinations = len(self.discreteParameters)
        # 0 0.5 1  1.5 2   2.5  3    3.5 4

    def first_iteration(self) -> list[SearchDataItem]:
        r"""
        The method performs the first iteration of the Global Search Algorithm
        """
        self.iterations_count = 1
        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        left = SearchDataItem(Point(self.evolvent.get_image(0.0), self.discreteParameters[0]), 0.0,
                              function_values=[FunctionValue()] * self.numberOfAllFunctions)
        image_right = self.evolvent.get_image(1.0)
        right: list[SearchDataItem] = []

        # [(x + y - 1)/y]

        items: list[SearchDataItem] = []
        image_x: list = []
        is_init_image_x: bool = False

        number_of_points_in_one_interval = \
            int(math.modf((self.parameters.number_of_parallel_points + self.numberOfParameterCombinations - 1)
                          / self.numberOfParameterCombinations)[1])

        h: float = 1.0 / (number_of_points_in_one_interval + 1)

        if self.parameters.start_point:
            for id_comb in range(self.numberOfParameterCombinations):
                if np.array_equal(self.parameters.start_point.discrete_variables, self.discreteParameters[id_comb]):
                    num_temp = number_of_points_in_one_interval - 1

                    ystart_point = Point(copy.copy(self.parameters.start_point.float_variables),
                                         self.discreteParameters[id_comb])
                    xstart_point = id_comb + self.evolvent.get_inverse_image(
                        self.parameters.start_point.float_variables)
                    itemstart_point = SearchDataItem(ystart_point, xstart_point, discrete_value_index=id_comb,
                                                     function_values=[FunctionValue()] * self.numberOfAllFunctions)

                    is_add_start_point: bool = False

                    for i in range(num_temp):
                        x = id_comb + h * (i + 1)

                        y_temp = self.evolvent.get_image(x)

                        y = Point(copy.copy(y_temp), self.discreteParameters[id_comb])
                        item = SearchDataItem(y, x, discrete_value_index=id_comb,
                                              function_values=[FunctionValue()] * self.numberOfAllFunctions)
                        if x < xstart_point < id_comb + h * (i + 1):
                            items.append(item)
                            items.append(itemstart_point)
                            is_add_start_point = True
                        else:
                            items.append(item)

                    if not is_add_start_point:
                        items.append(itemstart_point)

                else:
                    for i in range(number_of_points_in_one_interval):
                        x = id_comb + h * (i + 1)
                        if not is_init_image_x:
                            image_x.append(self.evolvent.get_image(x))

                        y = Point(copy.copy(image_x[i]), self.discreteParameters[id_comb])
                        item = SearchDataItem(y, x, discrete_value_index=id_comb,
                                              function_values=[FunctionValue()] * self.numberOfAllFunctions)
                        items.append(item)

                right.append(SearchDataItem(Point(copy.copy(image_right), self.discreteParameters[id_comb]),
                                            float(id_comb + 1),
                                            function_values=[FunctionValue()] * self.numberOfAllFunctions,
                                            discrete_value_index=id_comb))

                if not is_init_image_x:
                    is_init_image_x = True
        else:
            for id_comb in range(self.numberOfParameterCombinations):
                for i in range(number_of_points_in_one_interval):
                    x = id_comb + h * (i + 1)
                    if not is_init_image_x:
                        image_x.append(self.evolvent.get_image(x))

                    y = Point(copy.copy(image_x[i]), self.discreteParameters[id_comb])
                    item = SearchDataItem(y, x, discrete_value_index=id_comb,
                                          function_values=[FunctionValue()] * self.numberOfAllFunctions)
                    items.append(item)

                right.append(SearchDataItem(Point(copy.copy(image_right), self.discreteParameters[id_comb]),
                                            float(id_comb + 1),
                                            function_values=[FunctionValue()] * self.numberOfAllFunctions,
                                            discrete_value_index=id_comb))

                if not is_init_image_x:
                    is_init_image_x = True

        self.calculator.calculate_functionals_for_items(items)

        for item in items:
            self.update_optimum(item)

        left.delta = 0
        # left надо для всех считать
        self.calculate_global_r(left, None)

        items[0].delta = self.calculate_delta(left, items[0], self.dimension)
        self.calculate_global_r(items[0], left)

        for id_comb in range(self.numberOfParameterCombinations):
            if id_comb > 0:
                # вычисление left
                index = id_comb * number_of_points_in_one_interval
                items[index].delta = self.calculate_delta(right[id_comb - 1], items[index], self.dimension)
                self.calculate_global_r(items[index], right[id_comb - 1])

            for id_item in range(1, number_of_points_in_one_interval):
                index = id_comb * number_of_points_in_one_interval + id_item
                items[index].delta = self.calculate_delta(items[index - 1], items[index], self.dimension)
                self.calculate_global_r(items[index], items[index - 1])
                self.calculate_m(items[index], items[index - 1])

            left_index = id_comb * number_of_points_in_one_interval + number_of_points_in_one_interval - 1
            right[id_comb].delta = self.calculate_delta(items[left_index], right[id_comb], self.dimension)
            self.calculate_global_r(right[id_comb], items[left_index])

        # вставить left  и right, потом middle
        self.search_data.insert_first_data_item(left, right[-1])

        for right_item in range(self.numberOfParameterCombinations):
            if right_item < self.numberOfParameterCombinations - 1:
                self.search_data.insert_data_item(right[right_item], right[-1])

            for id_item in range(number_of_points_in_one_interval):
                index = right_item * number_of_points_in_one_interval + id_item
                self.search_data.insert_data_item(items[index], right[right_item])

        self.recalcR = True
        self.recalcM = True
        self.iterations_count = len(items)
        self.search_data.solution.number_of_global_trials = len(items)

        return items

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
        newy = self.evolvent.get_image(newx - math.modf(newx)[1])
        new = copy.deepcopy(SearchDataItem(Point(newy, old.point.discrete_variables),
                                           newx, discrete_value_index=old.get_discrete_value_index(),
                                           function_values=[FunctionValue()] * self.numberOfAllFunctions))
        # Обновление числа испытаний
        self.search_data.solution.number_of_global_trials += 1

        return new, old

    @staticmethod
    def GetDiscreteParameters(problem: Problem) -> list:
        list_discrete_values = list(problem.discrete_variable_values)
        return list(itertools.product(*list_discrete_values))

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
                if other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                    other_point = other_point.get_left()
                else:
                    other_point = None
                    break
            if other_point is not None and other_point.get_index() >= 0 \
                    and other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                # print(index)
                m = abs(other_point.function_values[index].value - curr_point.get_z()) / \
                    self.calculate_delta(other_point, curr_point, self.dimension)

            # Ищем справа
            other_point = left_point.get_right()
            if other_point is not None and other_point is curr_point:  # возможно только при пересчёте M
                other_point = other_point.get_right()
            while (other_point is not None) and (other_point.get_index() < curr_point.get_index()):
                if other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                    other_point = other_point.get_right()
                else:
                    other_point = None
                    break

            if other_point is not None and other_point.get_index() >= 0 \
                    and other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                m = max(m, abs(curr_point.get_z() - other_point.function_values[index].value) / \
                        self.calculate_delta(curr_point, other_point, self.dimension))

        if m > self.M[index] or (self.M[index] == 1.0 and m > 1e-12):
            self.M[index] = m
            self.recalcR = True
