import sys
from datetime import datetime
from typing import List

import traceback

import scipy
from scipy.optimize import Bounds

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.local_optimizer import LocalOptimize
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType
from iOpt.trial import Point


class Process:
    """
    Класс Process скрывает внутреннюю имплементацию класса Solver.
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener]
                 ):
        """
        Конструктор класса Process

        :param parameters: Параметры решения задачи оптимизации.
        :param task: Обёртка решаемой задачи.
        :param evolvent: Развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param search_data: Структура данных для хранения накопленной поисковой информации.
        :param method: Метод оптимизации, проводящий поисковые испытания по заданным правилам.
        :param listeners: Список "наблюдателей" (используется для вывода текущей информации).
        """
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.search_data = search_data
        self.method = method
        self._listeners = listeners
        self._first_iteration = True

    def solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: Текущая оценка решения задачи оптимизации
        """

        startTime = datetime.now()

        try:
            while not self.method.check_stop_condition():
                self.do_global_iteration()

        except Exception:
            print('Exception was thrown')
            print(traceback.format_exc())

        if self.parameters.refine_solution:
            self.do_local_refinement(self.parameters.local_method_iteration_count)

        result = self.get_results()
        result.solving_time = (datetime.now() - startTime).total_seconds()

        for listener in self._listeners:
            status = self.method.check_stop_condition()
            listener.on_method_stop(self.search_data, self.get_results(), status)

        return result

    def do_global_iteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: Количество итераций глобального поиска
        """
        number_ = number
        doneTrials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.before_method_start(self.method)
            doneTrials = self.method.first_iteration()
            self._first_iteration = False
            number = number - 1

        for _ in range(number):
            newpoint, oldpoint = self.method.calculate_iteration_point()
            self.method.calculate_functionals(newpoint)
            self.method.update_optimum(newpoint)
            self.method.renew_search_data(newpoint, oldpoint)
            self.method.finalize_iteration()
            doneTrials = self.search_data.get_last_items(self.parameters.number_of_parallel_points * number_)

        for listener in self._listeners:
            listener.on_end_iteration(doneTrials, self.get_results())

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: Количество итераций локального поиска
        """
        try:
            local_method_iteration_count = number
            if number == -1:
                local_method_iteration_count = int(self.parameters.localMethodIterationCount)

            result = self.GetResults()
            # start_point = result.bestTrials[0].point.floatVariables

            local_solution = LocalOptimize(self.task,
                                           method="Hooke-Jeeves", start_point=result.bestTrials[0].point,
                                           max_calcs=local_method_iteration_count,
                                           args={"eps": self.parameters.eps / 100, "step_mult": 2,
                                                 "max_iter": local_method_iteration_count}
                                           )
            # scipy.optimize.minimize(self.problemCalculate, x0=start_point, method='Nelder-Mead',
            #                        options={'maxiter': local_method_iteration_count})
            # local_solution = LocalOptimize(LocalTaskWrapper(self.task, result.bestTrials[0].point.discreteVariables),
            #                               method="Nelder-Mead", start_point=start_point,
            #                               args={"options": {'maxiter': local_method_iteration_count}})

            if local_method_iteration_count > 0:
                result.best_trials[0].point.float_variables = local_solution["x"]

                point: SearchDataItem = SearchDataItem(result.best_trials[0].point,
                                                       self.evolvent.get_inverse_image(
                                                           result.best_trials[0].point.float_variables),
                                                       function_values=[FunctionValue()] *
                                                                       (self.task.problem.number_of_constraints +
                                                                       self.task.problem.number_of_objectives)
                                                       )

                number_of_constraints = self.task.problem.number_of_constraints
                for i in range(number_of_constraints):
                    point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)
                    point.function_values[i] = self.task.problem.calculate(point.point, point.function_values[i])
                    point.set_z(point.function_values[i].value)
                    point.set_index(i)
                    if point.get_z() > 0:
                        break
                point.function_values[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV,
                                                                            number_of_constraints)
                point.function_values[number_of_constraints] = \
                    self.task.problem.calculate(point.point, point.function_values[number_of_constraints])
                point.set_z(point.function_values[number_of_constraints].value)
                point.set_index(number_of_constraints)

                result.best_trials[0].function_values = point.function_values

            result.numberOfLocalTrials = local_solution["fev"]
        except Exception:
            print("Local Refinement is not possible")

    def get_results(self) -> Solution:
        """
        Метод возвращает лучшее найденное решение задачи оптимизации

        :return: Решение задачи оптимизации
        """
       return self.search_data.solution

    def save_progress(self, file_name: str) -> None:
        """
        Сохранение процесса оптимизации из файла

        :param file_name: имя файла
        """
        self.search_data.save_progress(file_name=file_name)

    def load_progress(self, file_name: str) -> None:
        """
        Загрузка процесса оптимизации из файла

        :param file_name: имя файла
        """
        self.search_data.load_progress(file_name=file_name)
        self.method.iterationsCount = self.search_data.get_count() - 2

        for ditem in self.search_data:
            if ditem.get_index() >= 0:
                self.method.update_optimum(ditem)

        self.method.recalc_m()
        self.method.recalc_all_characteristics()
        self._first_iteration = False

        for listener in self._listeners:
            listener.before_method_start(self.method)

    '''
    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        #self.__listeners.append(listener)
        pass
    '''
