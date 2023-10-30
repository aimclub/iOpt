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

class TypeOfParetoRelation(Enum): # проблемы с именованием!!!
    DOMINANT = 1
    NONCOMPARABLE = 0 # NONCOMPARABLE
    NONDOMINATED = -1 #NONDOMINATED

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

            #Добавить вычисление свертки
            point = self.task.calculate(point, number_of_constraints+self.task.problem.number_of_objectives,
                                        TypeOfCalculation.CONVOLUTION) # по идее, там должен установиться z, а индекс не нужен (можно -1???)
            point.set_index(number_of_constraints)

        except Exception:
            point.set_z(sys.float_info.max)
            point.set_index(-10)

        return point

    def recalc_all_convolution(self) -> None:
        if self.is_recalc_all_convolution is not True:
            return
        for item in self.search_data:
            self.task.calculate(item, -1, TypeOfCalculation.CONVOLUTION) # индекс интовый, он ни на что не влияет
        self.is_recalc_all_convolution = False

        self.recalcR = True
        self.recalcM = True

        self.Z[self.task.problem.number_of_constraints] = self.best.get_z() #???? а точно ли?? лучшая же может поменяться
        # не логичнее ли запускать в update_optimum после смены min_max?



        #поднять флаги M и R
        # сменить Z (как?)
    # По влагу необходимо пересчитать все свертки (ВСЕ?! почему мн.ч), затем все R и перрезаполнить очередь (А нужно ли?!, если R меняются в предке)

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

        # добавить обновление
        # self.task.min_value
        # self.task.max_value
        # из IndexMethod

        # Оптимум поддерживает актуальное состоение найденой области Парето, а не одной точки!

        # НУЖНО ИЗМЕНИТЬ!
        #
        #


        #сделать копию, потом в конце присвоить ссылку начальному

        #(point.get_index()==number_of_constraints )

        #self.best = self.search_data.solution.best_trials[0] (если существует)

        if self.best is None or self.best.get_index() < point.get_index() or (
                self.best.get_index() == point.get_index() and point.get_z() < self.best.get_z()):
            self.best = point
            self.recalcR = True
            self.Z[point.get_index()] = point.get_z()

        # update_min_max_value
        # check_dominance
        #
        #self.search_data.solution.best_trials[0] = self.best меняется вся область Парето

        # может логичнее сначала изменить мин мах, потом пересчитать свертки, затем изменить бест,
        # потом уже область парето, потому что там ничего не зависит от get_z



    def check_dominance(self, point: SearchDataItem) -> None:
        r"""
         основная идея:
         пройти по всем найденным оптимальным триалам (точка и значение функций)
         сравнить отношение новой точки и точки из оптимальных
         если эта точка не лучше ни одной ни по одному критерию - забить на нее
         если она лучше по всем критериям какой-то точки - то выкинуть эту точку из оптимальных
         если она где-то лучше, где-то хуже текущих - добавить ее в список

        """
        pareto_front = copy.deepcopy(self.search_data.solution.best_trials)

        new = point.function_values
        add_point = 0
        for trial in pareto_front:
            old = trial.function_values
            relation = self.type_of_pareto_relation(new, old)
            if(relation == TypeOfParetoRelation.NONCOMPARABLE):
                # добавить новый в список, но нужно проверить, что он не добавлен?!
                add_point = 1
                # но нужно проверить, не доминируется ли кем-то еще,
            elif(relation == TypeOfParetoRelation.DOMINANT):
                # удалить старый и добавить новый
                add_point = 1
                pareto_front = np.delete(pareto_front, old) # насколько это вообще оптимально? Можно ли по другому? .. возвращает новый массив
            elif (relation == TypeOfParetoRelation.NONDOMINATED): # если доминируется хоть кем-то, то точно не добавляем
                add_point = 0
                break
            # если она хуже хотя бы одной - то выходим из рассмотрения вообще
        if add_point:
            pareto_front = np.append(pareto_front, new)
            # что тут нужно еще сделать?!

        self.search_data.solution.best_trials = pareto_front # вроде это и есть присвоение адреса

        # не стала делать заполнение с нуля, потому что мне кажется это не оптимальным:
        # если несравнимы, то добавляем оба
        # если новый доминирует, то добавляем только его
        # если новый кем-то доминируем, то придется с нуля заполнять массив
        # потому что если он раньше был с кем-то несравним, то его придется удалить
        # можно попробовать сделать это через set, но уйдет больше памяти на них
        # хотя снизу что-то получилось


        # способ 2
        pareto_front = np.ndarray(shape=(1), dtype=Trial)

        new = point.function_values
        add_point = 0
        for trial in self.search_data.solution.best_trials:
            old = trial.function_values
            relation = self.type_of_pareto_relation(new, old)
            if (relation == TypeOfParetoRelation.NONCOMPARABLE):
                # добавить новый в список, но нужно проверить, что он не добавлен?!
                add_point = 1
                pareto_front = np.append(pareto_front, old)
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
            pareto_front = np.append(pareto_front, new)
            self.search_data.solution.best_trials = pareto_front  # вроде это и есть присвоение адреса
        # если мы не добавляем точку, значит оставляем все как есть и ничего не меняем
            # что тут нужно еще сделать?!



    def type_of_pareto_relation(self, p1: np.ndarray(shape=(1), dtype=FunctionValue),
                                p2: np.ndarray(shape=(1), dtype=FunctionValue)) -> TypeOfParetoRelation:
        count_dom = 0
        number_of_objectives = self.task.problem.number_of_objectives
        for i in number_of_objectives:
            if(p1[i]<p2[i]):
                count_dom += 1
        if count_dom == 0 :
            return TypeOfParetoRelation.NONDOMINATED
        elif count_dom == number_of_objectives:
            return TypeOfParetoRelation.DOMINANT
        else:
            return TypeOfParetoRelation.NONCOMPARABLE
    def update_min_max_value(self,
                           data_item: SearchDataItem):
        # а где его применять?!
        if self.task.min_value and self.max_value: # проверка на пустоту
            for i in range(0, self.task.problem.number_of_objectives):
                if self.task.min_value[i] > data_item.function_values[i]:
                    self.task.min_value[i] = data_item.function_values[i]
                    self.is_recalc_all_convolution = True
                if self.task.max_value[i] < data_item.function_values[i]:
                    self.task.max_value[i] = data_item.function_values[i]
                    self.is_recalc_all_convolution = True # а нужно ли, если мах не влияет на свертку?
        else:
            self.task.min_value = data_item.function_values
            self.task.max_value = data_item.function_values
