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
        self.task.get_name()
        # Флаг используется при
        # 1. Запуске новой задачи (продолжение вычислений с новой сверткой)
        # 2. Обновлении минимума и максимума одного из критериев
        # По влагу необходимо пересчитать все свертки, затем все R и перрезаполнить очередь (А нужно ли?!, если R меняются в предке)
        self.is_recalc_all_convolution = True



    def calculate_functionals(self, point: SearchDataItem) -> SearchDataItem:
        r"""
        Проведение поискового испытания в заданной точке.

        :param point: точка, в которой надо провести испытание.

        :return: точка, в которой сохранены результаты испытания.
        """
        # из IndexMethod
        # Вычисляются ВСЕ критерии
        # Добавить вычисление свертки
        # Желательно использовать одну реализацию из MultiObjectiveMethodCalculator и не дублировать код

        # point это не точка, а SearchDataItem

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


            # вот как раз из-за этого не получается просто вынести метод в отдельный калькулятор целиком
            #if self.iterations_count == 1:
                #self.update_min_max_value(point) # сделать проверку на первую итерацию или куда-то вынести?!

            #Добавить вычисление свертки
            point = self.task.calculate(point, -1, TypeOfCalculation.CONVOLUTION)
            point.set_index(number_of_constraints)

        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-11)

        return point

    def recalc_all_convolution(self) -> None:
        if self.is_recalc_all_convolution is not True:
            return
        #пересчитать бест
        for item in self.search_data:
            self.task.calculate(item, -1, TypeOfCalculation.CONVOLUTION) # вынести в два цикла
            if(item.get_z()<self.best.get_z() and item.get_z()>0 and item.get_index()>=0): # достаточно ли неотрицательный индекс, или нужно равенство количеству ограничений # индекс должен быть больше или равен, чем у текущего
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

        #print(self.task.min_value)

        if self.best is None or self.best.get_index() < point.get_index() or (
                    self.best.get_index() == point.get_index() and point.get_z() < self.best.get_z()):
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()

        #print("point.get_index() in uo", point.get_index())

        if self.search_data.get_count() == 0: # или if self.iterations_count == 1: (наверно хуже, потому что считывание из файла заполняет SD и идет до update_optimum
            self.search_data.solution.best_trials[0] = self.best # на первой итерации нам нужно засунуть в лучшее хоть что-то


        if (point.get_index() == self.task.problem.number_of_constraints):  # а вот нужно ли его на весь блок кода или только эту часть?!
            self.update_min_max_value(point)
            if(self.search_data.get_count()>0): # см выше комментарий
                #self.recalc_all_convolution() # просто поднять флаг
                self.check_dominance(point)

        # вывод для проверки
        i = 0
        for trial in self.search_data.solution.best_trials:
            fv = [trial.function_values[k].value for k in range(self.task.problem.number_of_objectives)]
            print(fv, trial.point.float_variables)
            #print(i, trial.function_values[0].value, trial.function_values[1].value, trial.point.float_variables)
            i +=1

        print(self.best.get_z(), self.best.point.float_variables, self.best.function_values[0].value, self.best.function_values[1].value  )

    def check_dominance(self, point: SearchDataItem) -> None:
        pareto_front: np.ndarray(shape=(1), dtype=Trial) = []

        new = point.function_values # new - массив fv
        add_point = 0
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
            if(p1[i].value<p2[i].value): # нужно подумать над равенством
                count_dom += 1
            elif (p1[i].value == p2[i].value):
                count_equal += 1
        if count_dom == 0 :
            return TypeOfParetoRelation.NONDOMINATED
        elif (count_dom + count_equal) == number_of_objectives:
            return TypeOfParetoRelation.DOMINANT
        else:
            return TypeOfParetoRelation.NONCOMPARABLE
    def update_min_max_value(self,
                           data_item: SearchDataItem):
        #print("up1 ", self.task.min_value)
        # а где его применять?!
        if (self.task.min_value[0]==self.task.max_value[0] and self.task.min_value[0]==0):
            self.task.min_value = [fv.value for fv in data_item.function_values]
            self.task.max_value = [fv.value for fv in data_item.function_values]
        elif self.task.min_value and self.task.max_value: # проверка на пустоту
            for i in range(0, self.task.problem.number_of_objectives):
                if self.task.min_value[i] > data_item.function_values[i].value:
                    self.task.min_value[i] = data_item.function_values[i].value
                    self.is_recalc_all_convolution = True
                if self.task.max_value[i] < data_item.function_values[i].value:
                    self.task.max_value[i] = data_item.function_values[i].value
                    self.is_recalc_all_convolution = True # а нужно ли, если мах не влияет на свертку?
        # elif (self.task.min_value[0]==self.task.max_value[0]):
        #     self.task.min_value = [fv.value for fv in data_item.function_values]
        #     self.task.max_value = [fv.value for fv in data_item.function_values]

        #print("up2 ",self.task.min_value)



