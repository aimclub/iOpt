from typing import List
from datetime import datetime

import traceback

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.local_optimizer import local_optimize
from iOpt.method.method import Method
from iOpt.method.multi_objective_optim_task import MultiObjectiveOptimizationTask, MinMaxConvolution
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType
from iOpt.method.process import Process


class MCOProcess(Process):
    """
    Класс MCOProcess скрывает внутреннюю имплементацию класса Solver.
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener],
                 lambdas=[]
                 ):
        # if lambdas:
        #     self.lambdas = lambdas
        # elif parameters.start_lambdas:
        #     self.lambdas = parameters.start_lambdas
        # else:
        #     self.lambdas = [1.0 / task.problem.number_of_objectives] * task.problem.number_of_objectives
        # self.convolution = MinMaxConvolution(self.task.problem, self.lambdas)
        # self.base_task = task
        # mco_task = MultiObjectiveOptimizationTask(self.base_task.problem, self.convolution)
        # super().__init__(parameters, mco_task, evolvent, search_data, method, listeners)

        super().__init__(parameters, task, evolvent, search_data, method, listeners)

        self.number_of_lambdas = parameters.number_of_lambdas  # int
        if lambdas:
            self.start_lambdas = lambdas
        elif parameters.start_lambdas:
            self.start_lambdas = parameters.start_lambdas
        else:
            self.start_lambdas = [1.0 / task.problem.number_of_objectives] * task.problem.number_of_objectives

        self.current_lambdas = self.start_lambdas
        self.current_num_lambda = 0  # int
        self.lambdas_list = []  # список всех рассматриваемых
        self.lambdas_list.append(self.start_lambdas)
        # мб сделать мапу: лямбда-кол-во итераций? Или мало смысла?
        self.iterations_list = []

        self.convolution = MinMaxConvolution(self.task.problem, self.start_lambdas)
        self.base_task = task
        self.task = MultiObjectiveOptimizationTask(self.base_task.problem, self.convolution)
        self.method.task = self.task  # метод работает со своей собственной task, поэтому нужно заменить на правильную

        self.init_lambdas()

    def solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: Текущая оценка решения задачи оптимизации
        """

        start_time = datetime.now()

        try:
            for i in range(self.number_of_lambdas):
                while not self.method.check_stop_condition():
                    self.do_global_iteration()
                self.change_lambdas()
                print("self.iterations_list", self.iterations_list, "self.lambdas_list", self.lambdas_list)


        except Exception:
            print('Exception was thrown')
            print(traceback.format_exc())

        if self.parameters.refine_solution:
            self.do_local_refinement(self.parameters.local_method_iteration_count)

        result = self.get_results()
        result.solving_time = (datetime.now() - start_time).total_seconds()

        for listener in self._listeners:
            status = self.method.check_stop_condition()
            listener.on_method_stop(self.search_data, self.get_results(), status)

        return result

    def change_lambdas(self) -> None:
        self.method.min_delta = 1
        self.current_num_lambda += 1
        if self.current_num_lambda < self.number_of_lambdas:
            self.current_lambdas = self.lambdas_list[self.current_num_lambda] # вообще излишне знать текущую, если знаем номер
            self.task.convolution.lambda_param = self.current_lambdas
            #self.method.task = self.task # а нужно ли? пусть будет пока #или сделать какой-то метод у таски, который меняет лямбду # удаление этого кода ни на что не влияет (вроде)

            self.method.is_recalc_all_convolution = True
            if self.current_num_lambda == self.number_of_lambdas-1:
                self.iterations_list.append(self.method.convolution_iteration_count) # сохраняем, сколько по факту сделали итераций
                self.method.max_iter_for_convolution = self.parameters.global_method_iteration_count - \
                                                   self.method.iterations_count #добиваем весь остаток
                self.method.convolution_iteration_count = 0
                # проблема - не сохраняется количество итераций последнего набора лямбд
            else:
                # self.current_lambdas = self.lambdas_list[
                #     self.current_num_lambda]  # вообще излишне знать текущую, если знаем номер
                self.iterations_list.append(self.method.convolution_iteration_count)  # сохраняем, сколько по факту сделали итераций
                prev_remains = self.method.max_iter_for_convolution-self.method.convolution_iteration_count # план - факт
                self.method.max_iter_for_convolution = int(self.parameters.global_method_iteration_count /
                                                        self.number_of_lambdas) + prev_remains
                self.method.convolution_iteration_count = 0


    def init_lambdas(self) -> None:
        # TODO: разные способы инициализации лямбд, в тч для разных размерностей
        #self.method.min_delta = 1
        # первые лямбды уже лежат, нужно положить со вторых
        if self.task.problem.number_of_objectives == 2:  # пока только двумерный случай
            ndigits = 4 #количество цифр после запятой для округления
            if self.number_of_lambdas > 1:
                h = 1.0/(self.number_of_lambdas-1)
            else:
                h = 1
            #if (self.lambdas_list != []):
            prev_lambdas = self.start_lambdas#self.lambdas_list[0]
            # else:
            #     prev_lambdas = [0, 1]
            #     self.lambdas_list.append(prev_lambdas)
            for i in range(1, self.number_of_lambdas):
                l0 = self.start_lambdas[0] + i*h
                if l0 > 1:
                    l0 = l0 - 1
                l1 = 1 - l0
                lambdas = [l0, l1]
                self.lambdas_list.append(lambdas)
                prev_lambdas = lambdas


            # for i in range(1, self.number_of_lambdas):
            #     #l0 = prev_lambdas[0]+h if prev_lambdas[0]+h<1 else prev_lambdas[0]+h-1
            #     l0 = self.start_lambdas[0] + i*h if self.start_lambdas[0] + i*h < 1 else self.start_lambdas[0] + i*h - 1
            #     l0 = round(l0, ndigits)
            #     l1 = 1 - l0
            #     l1 = round(l1, ndigits)
            #     lambdas = [l0, l1]
            #     self.lambdas_list.append(lambdas)
            #     prev_lambdas = lambdas

            self.method.max_iter_for_convolution = int(self.parameters.global_method_iteration_count/self.number_of_lambdas)

            #self.iterations_list.append(self.method.iter_for_convolution)

            #self.task.convolution.lambda_param = self.lambdas_list[0]

            print("lambdas_list in process", self.lambdas_list)

    # остальные методы на данном этапе можно вызывать из родительского класса
    # TODO: проверить load/store
