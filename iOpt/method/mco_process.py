import copy
from typing import List
from datetime import datetime

import traceback

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.multi_objective_optim_task import MultiObjectiveOptimizationTask, MinMaxConvolution
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
                 task: MultiObjectiveOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener],
                 lambdas=[]
                 ):

        super().__init__(parameters, task, evolvent, search_data, method, listeners)

        self.number_of_lambdas = parameters.number_of_lambdas  # int
        if lambdas:
            self.start_lambdas = lambdas
        elif parameters.start_lambdas:
            self.start_lambdas = parameters.start_lambdas
        else:
            self.start_lambdas = []
        # else:
        #     self.start_lambdas = [[1.0 / task.problem.number_of_objectives] * task.problem.number_of_objectives]

        self.current_num_lambda = 0  # int
        self.lambdas_list = []  # список всех рассматриваемых
        self.iterations_list = []

        self.convolution = task.convolution
        self.task = task

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
            self.method.is_recalc_all_convolution = True

            self.iterations_list.append(self.method.iterations_count) # здесь будет накапливаться сумма итераций
            self.method.max_iter_for_convolution = int((self.parameters.global_method_iteration_count /
                                                       self.number_of_lambdas)*(self.current_num_lambda+1))


    def init_lambdas(self) -> None:
        if self.task.problem.number_of_objectives == 2:  # двумерный случай
            if self.number_of_lambdas > 1:
                h = 1.0/(self.number_of_lambdas-1)
            else:
                h = 1
            if not self.start_lambdas:
                for i in range(self.number_of_lambdas):
                    l0 = i * h
                    if l0 > 1:
                        l0 = l0 - 1
                    l1 = 1 - l0
                    lambdas = [l0, l1]
                    self.lambdas_list.append(lambdas)
            elif len(self.start_lambdas)==self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            elif len(self.start_lambdas)==1:
                self.lambdas_list.append(self.start_lambdas[0])
                for i in range(1, self.number_of_lambdas):
                    l0 = self.start_lambdas[0][0] + i*h
                    if l0 > 1:
                        l0 = l0 - 1
                    l1 = 1 - l0
                    lambdas = [l0, l1]
                    self.lambdas_list.append(lambdas)
            #else:
        else: # многомерный случай
            if len(self.start_lambdas)==self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            else:
                #  пример в методе, точность M - 10
                # #завести развертку размерности числа критериев, границы от 0 до 1, для формирования 10 лямбда: выбрать равномерно числа на отрезке от 0 до 1, после получения прообраза многомерной области суммировать получившиеся у и поделить каждый у на сумму(чтобы были в сумме 1),
                # округление не нужно
                if self.number_of_lambdas > 1:
                    h = 1.0 / (self.number_of_lambdas - 1)
                else:
                    h = 1
                evolvent = Evolvent([0]*self.task.problem.number_of_objectives, [1]*self.task.problem.number_of_objectives, self.task.problem.number_of_objectives)

                for i in range(self.number_of_lambdas):
                    x = i*h
                    y = evolvent.get_image(x)
                    sum = 0
                    for i in range(self.task.problem.number_of_objectives):
                        sum += y[i]
                    for i in range(self.task.problem.number_of_objectives):
                        y[i] = y[i] / sum
                    lambdas = list(y)
                    self.lambdas_list.append(lambdas)

        self.current_lambdas = self.lambdas_list[0]
        self.method.max_iter_for_convolution = int(
            self.parameters.global_method_iteration_count / self.number_of_lambdas)

    # TODO: проверить load/store

