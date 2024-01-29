import copy
from typing import Tuple

import math
import sys

import numpy as np

from enum import Enum
from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.multi_objective_optim_task import MultiObjectiveOptimizationTask
from iOpt.method.search_data import SearchDataItem, SearchData
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType, Trial
from iOpt.method.optim_task import TypeOfCalculation
from iOpt.method.method import Method

class TypeOfParetoRelation(Enum):
    DOMINANT = 1
    NONCOMPARABLE = 0
    NONDOMINATED = -1

class MultiObjectiveMethod(MixedIntegerMethod):
    """
    Класс Method содержит реализацию Алгоритма Глобального Поиска
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: MultiObjectiveOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData):
        super().__init__(parameters, task, evolvent, search_data)
        # Флаг используется при
        # 1. Запуске новой задачи (продолжение вычислений с новой сверткой)
        # 2. Обновлении минимума и максимума одного из критериев
        # По влагу необходимо пересчитать все свертки, затем все R и перрезаполнить очередь (А нужно ли?!, если R меняются в предке)
        self.is_recalc_all_convolution = True
        self.max_iter_for_convolution = 0
        self.convolution_iteration_count = 0

        self.number_of_lambdas = parameters.number_of_lambdas # int
        self.current_lambdas = parameters.start_lambdas # list double[dimension]
        self.current_num_lambda = 0 # int
        self.lambdas_list = [] # список всех рассматриваемых
        if(parameters.start_lambdas):
            self.lambdas_list.append(parameters.start_lambdas)





    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """

        # Желательно использовать одну реализацию из MultiObjectiveMethodCalculator и не дублировать код


        try:
            number_of_constraints = self.task.problem.number_of_constraints
            for i in range(number_of_constraints): # проходим по всем ограничениям
                point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)  # ???
                point = self.task.calculate(point, i) # ??? типа считаем, что перестановок нет и индекс соответствует индексу ограничения?? И сначала ограничения, потом критерии
                point.set_z(point.function_values[i].value)
                point.set_index(i)
                if point.get_z() > 0:
                    return point

            # Вычисляются ВСЕ критерии
            for i in range(self.task.problem.number_of_objectives):  # проходим по всем критериям
                point.function_values[number_of_constraints+i] = FunctionValue(FunctionType.OBJECTIV,
                                                                             number_of_constraints+i)
                point = self.task.calculate(point, number_of_constraints+i)

            # лучше сделать все-таки first_iteration..
            # if self.search_data.get_count() == 0:
            #     self.init_lambdas()

            #Добавить вычисление свертки
            point = self.task.calculate(point, -1, TypeOfCalculation.CONVOLUTION)
            print("point.get_z ", point.get_z())
            point.set_index(number_of_constraints)

        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-11)

        return point

    def recalc_all_convolution(self) -> None:
        print("recalc_all_convolution")
        if self.is_recalc_all_convolution is not True:
            return

        for item in self.search_data:
            self.task.calculate(item, -1, TypeOfCalculation.CONVOLUTION)

        # пересчитать бест
        if self.best:
            self.task.calculate(self.best, -1, TypeOfCalculation.CONVOLUTION)
        for item in self.search_data:
            if(item.get_z()<self.best.get_z() and item.get_z()>0 and item.get_index()>=self.best.get_index()):
                self.best = item

        self.is_recalc_all_convolution = False

        self.recalcR = True
        self.recalcM = True

        if self.best:
            self.Z[self.task.problem.number_of_constraints] = self.best.get_z()

    def calculate_iteration_point(self) -> Tuple[SearchDataItem, SearchDataItem]:  # return  (new, old)
        #обновление всех сверток по флагу self.is_recalc_all_convolution
        if self.is_recalc_all_convolution is True:
            self.recalc_all_convolution()

        return super(MultiObjectiveMethod, self).calculate_iteration_point()


    def update_optimum(self, point: SearchDataItem) -> None:
        r"""
        Обновляет оценку оптимума.

        :param point: точка нового испытания.
        """
        if self.best:
            print(self.best.get_index(), point.get_index(), self.best.get_z(), point.get_z())
            # 0.09939725612728388 0 0.09939725612728388 0
            # как раз здесь проблема в том, что из-за отсутствия мин мах у первой точки z получается нулевым и
            # 0 меньше, чем у best, поэтому best затирается
        #TODO: проверить, зачем это нужно и что будет, если убрать

        # if (self.best and self.best.get_z()>0 and self.iterations_count==1): # это работает, но выглядит плохо
        #     self.update_min_max_value(point)
        #     return
        if self.best is None or self.best.get_index() < point.get_index() or (
                    self.best.get_index() == point.get_index() and point.get_z() < self.best.get_z()):
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()

        if self.best:
            print("best 2", self.best.get_z(), self.best.point.float_variables, self.best.function_values[0].value, self.best.function_values[1].value  )
        print("point.get_index() in uo", point.get_index())

        if self.search_data.get_count() == 0: # или if self.iterations_count == 1: (наверно хуже, потому что считывание из файла заполняет SD и идет до update_optimum
            self.search_data.solution.best_trials[0] = self.best # на первой итерации нам нужно засунуть в лучшее хоть что-то


        if (point.get_index() == self.task.problem.number_of_constraints):  # а вот нужно ли его на весь блок кода или только эту часть?!
            self.update_min_max_value(point)
            self.check_dominance(point)

        # вывод для проверки
        i = 0
        for trial in self.search_data.solution.best_trials:
            fv = [trial.function_values[k].value for k in range(self.task.problem.number_of_objectives)]
            print(fv, trial.point.float_variables)
            #print(i, trial.function_values[0].value, trial.function_values[1].value, trial.point.float_variables)
            i +=1
        print("best ", self.best.get_z(), self.best.point.float_variables, self.best.function_values[0].value, self.best.function_values[1].value  )

    def check_dominance(self, point: SearchDataItem) -> None:
        if (self.search_data.get_count() == 0):
            return
        pareto_front: np.ndarray(shape=(1), dtype=Trial) = []

        new = point.function_values # new - массив fv
        add_point = 0 # мб интуитивнее было бы true/false
        for trial in self.search_data.solution.best_trials:
            old = trial.function_values
            relation = self.type_of_pareto_relation(new, old)
            if (relation == TypeOfParetoRelation.NONCOMPARABLE):
                # добавить новый в список, но нужно проверить, что он не добавлен?!
                add_point = 1
                pareto_front = np.append(pareto_front, trial) #состоит из трайлов, поэтому нужно засунуть трайл
                # но нужно проверить, не доминируется ли кем-то еще,
            elif (relation == TypeOfParetoRelation.DOMINANT):
                # удалить старый и добавить новый
                add_point = 1
            elif (relation == TypeOfParetoRelation.NONDOMINATED):  # если доминируется хоть кем-то,то точно не добавляем
                add_point = 0
                break
                # здесь не нужно добавлять старый, потому что если новый не добавляем, то ничгео не меняем
            # если она хуже хотя бы одной - то выходим из рассмотрения вообще
        print(add_point)
        if add_point:
            pareto_front = np.append(pareto_front, Trial(point.point, point.function_values)) #состоит из трайлов, поэтому нужно засунуть трайл
            self.search_data.solution.best_trials = pareto_front  # присвоение адреса
        # если мы не добавляем точку, значит оставляем все как есть и ничего не меняем


    def type_of_pareto_relation(self, p1: np.ndarray(shape=(1), dtype=FunctionValue),
                                p2: np.ndarray(shape=(1), dtype=FunctionValue)) -> TypeOfParetoRelation:
        count_dom = 0
        count_equal = 0
        number_of_objectives = self.task.problem.number_of_objectives
        for i in range(number_of_objectives):
            if (p1[i].value<p2[i].value):
                count_dom += 1
            elif (p1[i].value == p2[i].value):
                count_equal += 1
        if count_dom == 0:
            return TypeOfParetoRelation.NONDOMINATED
        elif (count_dom + count_equal) == number_of_objectives:
            return TypeOfParetoRelation.DOMINANT
        else:
            return TypeOfParetoRelation.NONCOMPARABLE
    def update_min_max_value(self,
                           data_item: SearchDataItem):
        if (self.task.min_value[0]==self.task.max_value[0] and self.task.min_value[0]==0):
            # мб имеет смысл это все вынести в инициализацию таски?
            if (len(self.search_data.solution.best_trials)>0): #если уже есть решения
                # пройти и найти мин мах
                self.task.min_value = [fv.value for fv in self.search_data.solution.best_trials[0].function_values]
                self.task.max_value = [fv.value for fv in self.search_data.solution.best_trials[0].function_values]
                for trial in self.search_data.solution.best_trials[1:]:
                    for i in range(0, self.task.problem.number_of_objectives):
                        if self.task.min_value[i] > trial.function_values[i].value:
                            self.task.min_value[i] = trial.function_values[i].value
                            self.is_recalc_all_convolution = True
                        if self.task.max_value[i] < trial.function_values[i].value:
                            self.task.max_value[i] = trial.function_values[i].value
                            self.is_recalc_all_convolution = True
            else:
                # поместить тот, что пришел
                self.task.min_value = [fv.value for fv in data_item.function_values]
                self.task.max_value = [fv.value for fv in data_item.function_values]
        #elif self.task.min_value and self.task.max_value: # проверка на пустоту
        else:
            for i in range(0, self.task.problem.number_of_objectives):
                if self.task.min_value[i] > data_item.function_values[i].value:
                    self.task.min_value[i] = data_item.function_values[i].value
                    self.is_recalc_all_convolution = True
                if self.task.max_value[i] < data_item.function_values[i].value:
                    self.task.max_value[i] = data_item.function_values[i].value
                    self.is_recalc_all_convolution = True

        print("up2 ",self.task.min_value, self.task.max_value)

    def check_stop_condition(self) -> bool:
        r"""
        Проверка условия остановки.
        Алгоритм должен завершить работу, когда достигнута точность eps или превышен лимит итераций.

        :return: True, если выполнен критерий остановки; False - в противном случае.
        """
        if self.min_delta < self.parameters.eps or self.convolution_iteration_count >= self.max_iter_for_convolution:
            self.stop = True
        else:
            self.stop = False

        return self.stop

    def finalize_iteration(self) -> None:
        r"""
        Заканчивает итерацию, обновляет счётчик итераций.
        """
        self.convolution_iteration_count += 1
        return super().finalize_iteration()


    def calculate_m(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        r"""
        Вычисление оценки константы Гельдера между между curr_point и left_point.

        :param curr_point: правая точка интервала
        :param left_point: левая точка интервала
        """
        # Обратить внимание на вычисление расстояния, должен использоваться метод CalculateDelta
        print("start calc_m")
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
            print("left_point.get_index() == index")
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
                # m = max(m, abs(curr_point.get_z() - other_point.function_values[index].value) / \
                #         self.calculate_delta(curr_point, other_point, self.dimension))
                #нужна проверка на индекс, если ограничение - то формула другая
                # TODO: изменить формулу для задач с ограничениями
                m = max(m, abs(curr_point.get_z() - other_point.get_z()) / \
                        self.calculate_delta(curr_point, other_point, self.dimension))

        if m > self.M[index] or (self.M[index] == 1.0 and m > 1e-12):
            self.M[index] = m
            self.recalcR = True
        print(f"index={index}, lp.i={left_point.get_index()}, lp.z={left_point.get_z()}, cp.z={curr_point.get_z()}")
        print("M in calc_m:", self.M)

# if __name__ == "__main__":
#     lambdas_list = [[0.319, 0.681]]
#     number_of_lambdas = 10
#     ndigits = 4
#
#     h = 1 / number_of_lambdas
#     #prev_lambdas = lambdas_list[0] if (lambdas_list!=[]) else [0, 1]
#     if (lambdas_list!=[]):
#         prev_lambdas = lambdas_list[0]
#     else:
#         prev_lambdas = [0, 1]
#         lambdas_list.append(prev_lambdas)
#     for i in range(number_of_lambdas - 1):
#         l0 = prev_lambdas[0] + h if prev_lambdas[0] + h < 1 else prev_lambdas[0] + h - 1
#         l0 = round(l0, ndigits) #округление, потому что иначе будет что-то типа: 0.7999999999999999 и
#         l1 = 1 - l0
#         l1 = round(l1, ndigits)
#         lambdas = [l0, l1]
#         lambdas_list.append(lambdas)
#         prev_lambdas = lambdas
#
#     print(lambdas_list)




