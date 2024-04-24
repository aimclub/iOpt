from typing import Tuple

import numpy as np

from enum import Enum
from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.mco_optim_task import MCOOptimizationTask
from iOpt.method.search_data import SearchDataItem, SearchData
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType, Trial
from iOpt.method.optim_task import TypeOfCalculation


class TypeOfParetoRelation(Enum):
    DOMINANT = 1
    NONCOMPARABLE = 0
    NONDOMINATED = -1


class MCOMethod(MixedIntegerMethod):
    """
    The MCOMethod class contains an implementation of the Global Search Algorithm for multi-criteria problems
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: MCOOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator):
        super().__init__(parameters, task, evolvent, search_data, calculator)
        self.is_recalc_all_convolution = True
        self.max_iter_for_convolution = 0

    def set_max_iter_for_convolution(self, max_iter_for_convolution) -> None:
        self.max_iter_for_convolution = max_iter_for_convolution

    def set_min_delta(self, min_delta) -> None:
        self.min_delta = min_delta
        self.is_recalc_all_convolution = True

    def recalc_all_convolution(self) -> None:
        if self.is_recalc_all_convolution is not True:
            return

        for item in self.search_data:
            self.task.calculate(item, -1, TypeOfCalculation.CONVOLUTION)

        if self.best:
            self.task.calculate(self.best, -1, TypeOfCalculation.CONVOLUTION)
        for item in self.search_data:
            if self.best.get_z() > item.get_z() > 0 and item.get_index() >= self.best.get_index():
                self.best = item

        self.is_recalc_all_convolution = False

        self.recalcR = True
        self.recalcM = True

        if self.best:
            self.Z[self.task.problem.number_of_constraints] = self.best.get_z()

    def calculate_iteration_point(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        if self.is_recalc_all_convolution is True:
            self.recalc_all_convolution()

        return super(MCOMethod, self).calculate_iteration_point()

    def update_optimum(self, point: SearchDataItem) -> None:
        r"""
        Updates the estimate of the optimum.

        :param point: new trial point.
        """
        if self.best is None or self.best.get_index() < point.get_index() or (
                self.best.get_index() == point.get_index() and point.get_z() < self.best.get_z()):
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()

        if not self.search_data.solution.best_trials[0].point:
            self.search_data.solution.best_trials[0] = self.best

        if point.get_index() == self.task.problem.number_of_constraints:
            self.update_min_max_value(point)
            self.pareto_set_update(point)

    def pareto_set_update(self, point: SearchDataItem) -> None:
        if self.search_data.get_count() == 0:
            return

        pareto_set: np.ndarray(shape=(1), dtype=Trial) = []
        new_point = point.function_values
        add_point = False

        for trial in self.search_data.solution.best_trials:
            old_point = trial.function_values
            relation = self.type_of_pareto_relation(new_point, old_point)
            if relation == TypeOfParetoRelation.NONCOMPARABLE:
                add_point = True
                pareto_set = np.append(pareto_set, trial)
            elif relation == TypeOfParetoRelation.DOMINANT:
                add_point = True
            elif relation == TypeOfParetoRelation.NONDOMINATED:
                add_point = False
                break
        if add_point:
            pareto_set = np.append(pareto_set, Trial(point.point, point.function_values))
            self.search_data.solution.best_trials = pareto_set
        # if we don't add a point, then the pareto set doesn't change.

    def type_of_pareto_relation(self, p1: np.ndarray(shape=(1), dtype=FunctionValue),
                                p2: np.ndarray(shape=(1), dtype=FunctionValue)) -> TypeOfParetoRelation:
        count_dom = 0
        count_equal = 0
        number_of_objectives = self.task.problem.number_of_objectives
        for i in range(number_of_objectives):
            if p1[i].value < p2[i].value:
                count_dom += 1
            elif p1[i].value == p2[i].value:
                count_equal += 1
        if count_dom == 0:
            return TypeOfParetoRelation.NONDOMINATED
        elif (count_dom + count_equal) == number_of_objectives:
            return TypeOfParetoRelation.DOMINANT
        else:
            return TypeOfParetoRelation.NONCOMPARABLE

    def update_min_max_value(self,
                             data_item: SearchDataItem):
        # If the minimum and maximum values have not yet been changed after initialization
        if self.task.min_value[0] == self.task.max_value[0] and self.task.min_value[0] == 0:
            # if the search information has been uploaded
            if self.search_data.get_count() > 0:
                self.task.min_value = [fv.value for fv in self.search_data.get_last_item().function_values]
                self.task.max_value = [fv.value for fv in self.search_data.get_last_item().function_values]
                for trial in self.search_data:
                    for i in range(0, self.task.problem.number_of_objectives):
                        if self.task.min_value[i] > trial.function_values[i].value:
                            self.task.min_value[i] = trial.function_values[i].value
                            self.is_recalc_all_convolution = True
                        if self.task.max_value[i] < trial.function_values[i].value:
                            self.task.max_value[i] = trial.function_values[i].value
                            self.is_recalc_all_convolution = True
            else:
                # assign the values of the first calculated item
                self.task.min_value = [fv.value for fv in data_item.function_values]
                self.task.max_value = [fv.value for fv in data_item.function_values]
        else:
            # comparing the value of the current min max with the values of the new point
            for i in range(0, self.task.problem.number_of_objectives):
                if self.task.min_value[i] > data_item.function_values[i].value:
                    self.task.min_value[i] = data_item.function_values[i].value
                    self.is_recalc_all_convolution = True
                if self.task.max_value[i] < data_item.function_values[i].value:
                    self.task.max_value[i] = data_item.function_values[i].value
                    self.is_recalc_all_convolution = True

    def check_stop_condition(self) -> bool:
        r"""
        Checking the stopping condition.
        The algorithm must terminate when the eps accuracy is reached or the iteration limit is exceeded.

        :return: True if the stopping criterion is met; False - otherwise.
        """
        if self.min_delta < self.parameters.eps or self.iterations_count >= self.max_iter_for_convolution:
            self.stop = True
        else:
            self.stop = False

        return self.stop

    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Compute the estimate of the Holder constant between curr_point and left_point.

        :param curr_point: right point of the interval
        :param left_point: left point of the interval
        """
        if curr_point is None:
            print("CalculateM: curr_point is None")
            raise RuntimeError("CalculateM: curr_point is None")
        if left_point is None:
            return
        index = curr_point.get_index()
        if index < 0:
            return
        m = 0.0
        if left_point.get_index() == index:
            m = abs(left_point.get_z() - curr_point.get_z()) / curr_point.delta
        else:
            other_point = left_point
            while (other_point is not None) and (other_point.get_index() < curr_point.get_index()):
                if other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                    other_point = other_point.get_left()
                else:
                    other_point = None
                    break
            if other_point is not None and other_point.get_index() >= 0 \
                    and other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                m = abs(other_point.function_values[index].value - curr_point.get_z()) / \
                    self.calculate_delta(other_point, curr_point, self.dimension)

            other_point = left_point.get_right()
            if other_point is not None and other_point is curr_point:
                other_point = other_point.get_right()
            while (other_point is not None) and (other_point.get_index() < curr_point.get_index()):
                if other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                    other_point = other_point.get_right()
                else:
                    other_point = None
                    break

            if other_point is not None and other_point.get_index() >= 0 \
                    and other_point.get_discrete_value_index() == curr_point.get_discrete_value_index():
                m = max(m, abs(curr_point.get_z() - other_point.get_z()) / \
                        self.calculate_delta(curr_point, other_point, self.dimension))

        if m > self.M[index] or (self.M[index] == 1.0 and m > 1e-12):
            self.M[index] = m
            self.recalcR = True
