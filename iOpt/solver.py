from typing import List
import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.index_method_evaluate import IndexMethodEvaluate
from iOpt.method.listener import Listener
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.solverFactory import SolverFactory
from iOpt.problem import Problem
from iOpt.routine.timeout import timeout
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


class Solver:
    """
    Класс Solver предназначен для выбора оптимальных (в заданной метрике) значений параметров
    сложных объектов и процессов, например, методов искусственного интеллекта и
    машинного обучения, а также – методов эвристической оптимизации.
    """

    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters = SolverParameters()
                 ):
        """
        Конструктор класса Solver

        :param problem: Постановка задачи оптимизации
        :param parameters: Параметры поиска оптимальных решений
        """

        self.problem = problem
        self.parameters = parameters

        Solver.check_parameters(self.problem, self.parameters)

        self.__listeners: List[Listener] = []

        self.search_data = SearchData(problem)
        self.evolvent = Evolvent(problem.lower_bound_of_float_variables, problem.upper_bound_of_float_variables,
                                 problem.number_of_float_variables)
        self.task = SolverFactory.create_task(problem, parameters)

        self.calculator = SolverFactory.create_calculator(self.task, self.parameters)

        self.method = SolverFactory.create_method(parameters, self.task, self.evolvent,
                                                  self.search_data, self.calculator)
        self.process = SolverFactory.create_process(parameters=parameters, task=self.task, evolvent=self.evolvent,
                                                    search_data=self.search_data, method=self.method,
                                                    listeners=self.__listeners, calculator=self.calculator)

    def solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: решение задачи оптимизации
        """
        Solver.check_parameters(self.problem, self.parameters)
        if self.parameters.timeout < 0:
            sol = self.process.solve()
        else:
            solv_with_timeout = timeout(seconds=self.parameters.timeout * 60)(self.process.solve)
            try:
                solv_with_timeout()
                sol = self.get_results()
            except Exception as exc:
                print(exc)
                sol = self.get_results()
                sol.solving_time = self.parameters.timeout * 60
                self.method.recalcR = True
                self.method.recalcM = True
                status = self.method.check_stop_condition()
                for listener in self.__listeners:
                    listener.on_method_stop(self.search_data, self.get_results(), status)
        return sol

    def do_global_iteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: число итераций глобального поиска
        """
        Solver.check_parameters(self.problem, self.parameters)
        self.process.do_global_iteration(number)

    def do_local_refinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: число итераций локального поиска
        """
        Solver.check_parameters(self.problem, self.parameters)
        self.process.do_local_refinement(number)

    def get_results(self) -> Solution:
        """
        Метод позволяет получить текущую оценку решения задачи оптимизации

        :return: решение задачи оптимизации
        """
        return self.process.get_results()

    def save_progress(self, file_name: str) -> None:
        """
        Сохранение процесса оптимизации в файл

        :param file_name: имя файла
        """
        self.process.save_progress(file_name=file_name)

    def load_progress(self, file_name: str) -> None:
        """
        Загрузка процесса оптимизации из файла

        :param file_name: имя файла
        """
        Solver.check_parameters(self.problem, self.parameters)
        self.process.load_progress(file_name=file_name)

        if (self.problem.number_of_discrete_variables > 0):
            self.process.method.iterations_count = self.process.search_data.get_count() - (len(self.method.GetDiscreteParameters(self.problem)) + 1 ) #-2
        else: self.process.method.iterations_count = self.process.search_data.get_count() - 2

    def refresh_listener(self) -> None:
        """
        Метод оповещения наблюдателей о произошедшем событии
        """

        pass

    def add_listener(self, listener: Listener) -> None:
        """
        Добавления наблюдателя за процессом оптимизации

        :param listener: объект класса реализующий методы наблюдения
        """

        self.__listeners.append(listener)

    @staticmethod
    def check_parameters(problem: Problem,
                         parameters: SolverParameters = SolverParameters()) -> None:
        """
        Проверяет параметры решателя

        :param problem: Постановка задачи оптимизации
        :param parameters: Параметры поиска оптимальных решений

        """

        if parameters.eps <= 0:
            raise Exception("search precision is incorrect, parameters.eps <= 0")
        if parameters.r <= 1:
            raise Exception("The reliability parameter should be greater 1. r>1")
        if parameters.iters_limit < 1:
            raise Exception("The number of iterations must not be negative. iters_limit>0")
        if parameters.evolvent_density < 2 or parameters.evolvent_density > 20:
            raise Exception("Evolvent density should be within [2,20]")
        if parameters.eps_r < 0 or parameters.eps_r >= 1:
            raise Exception("The epsilon redundancy parameter must be within [0, 1)")

        if problem.number_of_float_variables < 1:
            raise Exception("Must have at least one float variable")
        if problem.number_of_discrete_variables < 0:
            raise Exception("The number of discrete parameters must not be negative")
        if problem.number_of_objectives < 1:
            raise Exception("At least one criterion must be defined")
        if problem.number_of_constraints < 0:
            raise Exception("The number of сonstraints must not be negative")

        if len(problem.float_variable_names) != problem.number_of_float_variables:
            raise Exception("Floaf parameter names are not defined")

        if len(problem.lower_bound_of_float_variables) != problem.number_of_float_variables:
            raise Exception("List of lower bounds for float search variables defined incorrectly")
        if len(problem.upper_bound_of_float_variables) != problem.number_of_float_variables:
            raise Exception("List of upper bounds for float search variables defined incorrectly")

        for lower_bound, upper_bound in zip(problem.lower_bound_of_float_variables,
                                            problem.upper_bound_of_float_variables):
            if lower_bound >= upper_bound:
                raise Exception("For floating point search variables, "
                                "the upper search bound must be greater than the lower.")

        if problem.number_of_discrete_variables > 0:
            if len(problem.discrete_variable_names) != problem.number_of_discrete_variables:
                raise Exception("Discrete parameter names are not defined")

            for discreteValues in problem.discrete_variable_values:
                if len(discreteValues) < 1:
                    raise Exception("Discrete variable values not defined")

        if parameters.start_point:
            if len(parameters.start_point.float_variables) != problem.number_of_float_variables:
                raise Exception("Incorrect start point size")
            if parameters.start_point.discrete_variables:
                if len(parameters.start_point.discrete_variables) != problem.number_of_discrete_variables:
                    raise Exception("Incorrect start point discrete variables")
            for lower_bound, upper_bound, y in zip(problem.lower_bound_of_float_variables,
                                                   problem.upper_bound_of_float_variables,
                                                   parameters.start_point.float_variables):
                if y < lower_bound or y > upper_bound:
                    raise Exception("Incorrect start point coordinate")
        if parameters.number_of_lambdas:
            if parameters.number_of_lambdas < 0:
                raise Exception("Number of lambda sets is incorrect, parameters.number_of_lambdas <= 0")
        if parameters.start_lambdas:
            if len(parameters.start_lambdas)>1 and len(parameters.start_lambdas)<parameters.number_of_lambdas:
                raise Exception("The number of sets of initial lambdas must match the number of sets of lambdas "
                                "or be equal to 1")
            if not all(len(lambdas) == problem.number_of_objectives for lambdas in parameters.start_lambdas):
                raise Exception("The number of lambdas in the set must match the number of objectives for the problem")
            if not all(all(lamb >= 0 for lamb in lambdas) for lambdas in parameters.start_lambdas):
                raise Exception("The lambda parameters must be non-negative")


        # неортицательные и корличество или 1 или совпадет с кол-вом л


