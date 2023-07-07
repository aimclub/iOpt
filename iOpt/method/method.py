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
                 search_data: SearchData
                 ):
        r"""
        Конструктор класса Method

        :param parameters: параметры решения задачи оптимизации.
        :param task: обёртка решаемой задачи.
        :param evolvent: развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param search_data: структура данных для хранения накопленной поисковой информации.
        """
        self.stop: bool = False
        self.recalcR: bool = True
        self.recalcM: bool = True
        self.iterationsCount: int = 0
        self.best: SearchDataItem = None

        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.search_data = search_data
        # change to np.array, but indexing np is slower
        self.M = [1.0 for _ in range(task.problem.number_of_objectives + task.problem.number_of_constraints)]
        self.Z = [np.infty for _ in range(task.problem.number_of_objectives + task.problem.number_of_constraints)]
        self.dimension = task.problem.number_of_float_variables  # А ДЛЯ ДИСКРЕТНЫХ?
        self.search_data.solution.solution_accuracy = np.infty
        self.numberOfAllFunctions = task.problem.number_of_objectives + task.problem.number_of_constraints

    @property
    def min_delta(self):
        return self.search_data.solution.solution_accuracy

    @min_delta.setter
    def min_delta(self, val):
        self.search_data.solution.solution_accuracy = val

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

    def calculate_delta(self, l_point: SearchDataItem, r_point: SearchDataItem, dimension: int) -> float:
        """
        Вычисляет гельдерово расстояние в метрике Гельдера между двумя точками на отрезке [0,1],
          полученными при редукции размерности.

        :param l_point: левая точка
        :param r_point: правая точка
        :param dimension: размерность исходного пространства

        :return: гельдерово расстояние между lx и rx.
        """
        return pow(r_point.get_x() - l_point.get_x(), 1.0 / dimension)

    def first_iteration(self, calculator: Calculator = None) -> list[SearchDataItem]:
        r"""
        Метод выполняет первую итерацию Алгоритма Глобального Поиска.
        """

        # Генерация 3х точек 0, 0.5, 1. Значение функции будет вычисляться только в точке 0.5.
        # Интервал задаётся правой точкой, т.е. будут интервалы только для 0.5 и 1
        left = SearchDataItem(Point(self.evolvent.get_image(0.0), None), 0.,
                              function_values=[FunctionValue()] * self.numberOfAllFunctions)
        right = SearchDataItem(Point(self.evolvent.get_image(1.0), None), 1.0,
                               function_values=[FunctionValue()] * self.numberOfAllFunctions)

        items: list[SearchDataItem] = []

        if self.parameters.start_point:
            number_of_point: int = self.parameters.number_of_parallel_points - 1
            h: float = 1.0 / (number_of_point + 1)

            ystart_point = Point(copy.copy(self.parameters.start_point.float_variables), None)
            xstart_point = self.evolvent.get_inverse_image(self.parameters.start_point.float_variables)

            itemstart_point = SearchDataItem(ystart_point, xstart_point,
                                            function_values=[FunctionValue()] * self.numberOfAllFunctions)

            isAddstart_point: bool = False

            for i in range(number_of_point):
                x = h * (i + 1)
                y = Point(self.evolvent.get_image(x), None)
                item = SearchDataItem(y, x,
                                      function_values=[FunctionValue()] * self.numberOfAllFunctions)
                if x < xstart_point < h * (i + 2):
                    items.append(item)
                    items.append(itemstart_point)
                    isAddstart_point = True
                else:
                    items.append(item)

            if not isAddstart_point:
                items.append(itemstart_point)
        else:

            number_of_point: int = self.parameters.number_of_parallel_points
            h: float = 1.0 / (number_of_point + 1)

            for i in range(number_of_point):
                x = h * (i + 1)
                y = Point(self.evolvent.get_image(x), None)
                item = SearchDataItem(y, x,
                                      function_values=[FunctionValue()] * self.numberOfAllFunctions)
                items.append(item)

        if calculator is None:
            for item in items:
                self.calculate_functionals(item)
        else:
            calculator.calculate_functionals_for_items(items)

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
        # self.search_data.InsertDataItem(middle, right)

        for item in items:
            self.search_data.insert_data_item(item, right)

        self.recalcR = True
        self.recalcM = True

        self.iterationsCount = len(items)
        self.search_data.solution.number_of_global_trials = len(items)

        return items

    def check_stop_condition(self) -> bool:
        r"""
        Проверка условия остановки.
        Алгоритм должен завершить работу, когда достигнута точность eps или превышен лимит итераций.

        :return: True, если выполнен критерий остановки; False - в противном случае.
        """
        if self.min_delta < self.parameters.eps or self.iterationsCount >= self.parameters.global_method_iteration_count:
            self.stop = True
        else:
            self.stop = False

        return self.stop

    def recalc_m(self) -> None:
        r"""
        Пересчёт оценки константы Липшица.
        """
        if self.recalcM is not True:
            return
        for item in self.search_data:
            self.calculate_m(item, item.get_left())
        self.recalcM = False

    def recalc_all_characteristics(self) -> None:
        r"""
        Пересчёт характеристик для всех поисковых интервалов.
        """
        if self.recalcR is not True:
            return
        self.search_data.clear_queue()
        for item in self.search_data:  # Должно работать...
            self.calculate_global_r(item, item.get_left())
            # self.CalculateLocalR(item)
        self.search_data.refill_queue()
        self.recalcR = False

    def calculate_next_point_coordinate(self, point: SearchDataItem) -> float:
        r"""
        Вычисление точки нового испытания :math:`x^{k+1}` в заданном интервале :math:`[x_{t-1},x_t]`.

        :param point: интервал, заданный его правой точкой :math:`x_t`.

        :return: точка нового испытания :math:`x^{k+1}` в этом интервале.
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
        Вычисление точки нового испытания :math:`x^{k+1}`.

        :return: :math:`x^{k+1}` - точка нового испытания, и :math:`x_t` - левая точка интервала :math:`[x_{t-1},x_t]`,
          которому принадлежит :math:`x^{k+1}`, т.е. :math:`x^{k+1} \in [x_{t-1},x_t]`.
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
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        try:
            point = self.task.calculate(point, 0)
            point.set_z(point.function_values[0].value)
            point.set_index(0)
        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-10)

        return point

    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
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
        index = curr_point.get_index()
        if left_point.get_index() == index and index >= 0:  # А если не равны, то надо искать ближайший левый/правый с таким индексом
            m = abs(left_point.get_z() - curr_point.get_z()) / curr_point.delta
            if m > self.M[index]:
                self.M[index] = m
                self.recalcR = True

    # def CalculateM(self, point: SearchDataItem):  # В python нет такой перегрузки функций, надо менять название
    #     self.CalculateM(point, point.GetLeft())

    def calculate_global_r(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление глобальной характеристики интервала [left_point, curr_point].

        :param curr_point: правая точка интервала.
        :param left_point: левая точка интервала.
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
            globalR = 2 * deltax - 4 * math.fabs(self.Z[0]) / (r * self.M[0])
        elif left_point.get_index() == curr_point.get_index():
            v = curr_point.get_index()
            globalR = deltax + (zr - zl) * (zr - zl) / (deltax * self.M[v] * self.M[v] * r * r) - \
                      2 * (zr + zl - 2 * self.Z[v]) / (r * self.M[v])
        elif left_point.get_index() < curr_point.get_index():
            v = curr_point.get_index()
            globalR = 2 * deltax - 4 * (zr - self.Z[v]) / (r * self.M[v])
        else:
            v = left_point.get_index()
            globalR = 2 * deltax - 4 * (zl - self.Z[v]) / (r * self.M[v])
        curr_point.globalR = globalR

    def renew_search_data(self, newpoint: SearchDataItem, oldpoint: SearchDataItem) -> None:
        """
        Метод обновляет всю поисковую информацию: длины интервалов, константы Гёльдера, все характеристики и вставляет
          новую точку в хранилище.

        :param newpoint: новая точка
        :param oldpoint: правая точка интервала, которому принадлежит новая точка
        """

        # oldpoint.delta = Method.CalculateDelta(newpoint.GetX(), oldpoint.GetX(), self.dimension)
        # newpoint.delta = Method.CalculateDelta(oldpoint.GetLeft().GetX(), newpoint.GetX(), self.dimension)

        oldpoint.delta = self.calculate_delta(newpoint, oldpoint, self.dimension)
        newpoint.delta = self.calculate_delta(oldpoint.get_left(), newpoint, self.dimension)

        self.calculate_m(newpoint, oldpoint.get_left())
        self.calculate_m(oldpoint, newpoint)

        self.calculate_global_r(newpoint, oldpoint.get_left())
        self.calculate_global_r(oldpoint, newpoint)

        self.search_data.insert_data_item(newpoint, oldpoint)

    def update_optimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.

        :param point: точка нового испытания.
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
        Заканчивает итерацию, обновляет счётчик итераций.
        """
        self.iterationsCount += 1

    def get_iterations_count(self) -> int:
        r"""
        Возвращает число выполненных итераций.

        :return:  число выполненных итераций.
        """
        return self.iterationsCount

    def get_optimum_estimation(self) -> SearchDataItem:
        r"""
        Возвращает оценку оптимума.

        :return: текущая оценка оптимума.
        """
        return self.best
