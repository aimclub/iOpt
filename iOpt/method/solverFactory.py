from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.async_parallel_process import AsyncParallelProcess

from iOpt.method.calculator import Calculator
from iOpt.method.default_calculator import DefaultCalculator
from iOpt.method.index_method import IndexMethod
from iOpt.method.index_method_evaluate import IndexMethodEvaluate
from iOpt.method.listener import Listener
from iOpt.method.mco_method_many_lambdas import MCOMethodManyLambdas
from iOpt.method.method import Method
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.mco_method_evaluate import MCOMethodEvaluate
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.mco_optim_task import MCOOptimizationTask, MinMaxConvolution
from iOpt.problem import Problem
from iOpt.method.parallel_process import ParallelProcess
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters


class SolverFactory:
    """
    The SolverFactory class creates suitable solution method and process classes according to the given parameters
    """

    def __init__(self):
        pass

    @staticmethod
    def create_task(problem: Problem,
                    parameters: SolverParameters) -> OptimizationTask:
        """
        Создает подходящий класс метода решения по заданным параметрам

        :param parameters: параметры решения задачи оптимизации.
        :param task: обёртка решаемой задачи.
        :param evolvent: развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param search_data: структура данных для хранения накопленной поисковой информации.

        :return: созданный метод
        """
        if problem.number_of_objectives > 1:
            if parameters.start_lambdas:
                convolution = MinMaxConvolution(problem, parameters.start_lambdas[0], parameters.is_scaling)
            else:
                convolution = MinMaxConvolution(problem, [1.0 / problem.number_of_objectives] * problem.number_of_objectives, parameters.is_scaling)
            return MCOOptimizationTask(problem, convolution)
        else:
            return OptimizationTask(problem)


    @staticmethod
    def create_evaluate_method(task: OptimizationTask):
        if task.problem.number_of_objectives > 1:
            return MCOMethodEvaluate(task)
        else:
            return IndexMethodEvaluate(task)

    @staticmethod
    def create_calculator(task: OptimizationTask,
                          parameters: SolverParameters):
        index_method_evaluate = SolverFactory.create_evaluate_method(task)
        if parameters.number_of_parallel_points > 1:
            return Calculator(index_method_evaluate, parameters)
        else:
            return DefaultCalculator(index_method_evaluate, parameters)

    @staticmethod
    def create_method(parameters: SolverParameters,
                      task: OptimizationTask,
                      evolvent: Evolvent,
                      search_data: SearchData,
                      calculator: Calculator) -> Method:
        """
        Create a suitable solution method class based on the given parameters

        :param parameters: parameters of the solution of the optimization problem.
        :param task: the wrapper of the problem to be solved.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: data structure for storing accumulated search information.
        :param calculator: класс, содержащий методы проведения испытаний (параллельные и/или индексную схему)

        :return: created method
        """
        if task.problem.number_of_objectives > 1:
            return MCOMethodManyLambdas(parameters, task, evolvent, search_data, calculator)
        elif task.problem.number_of_discrete_variables > 0:
            return MixedIntegerMethod(parameters, task, evolvent, search_data, calculator)
        elif task.problem.number_of_constraints > 0:
            return IndexMethod(parameters, task, evolvent, search_data, calculator)
        else:
            return Method(parameters, task, evolvent, search_data, calculator)

    @staticmethod
    def create_process(parameters: SolverParameters,
                       task: OptimizationTask,
                       evolvent: Evolvent,
                       search_data: SearchData,
                       method: Method,
                       listeners: List[Listener],
                       calculator: Calculator) -> Process:
        """
        Create a suitable process class based on the specified parameters

        :param parameters: parameters of the solution of the optimization problem.
        :param task: the wrapper of the problem to be solved.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: data structure for storing accumulated search information.
        :param method: An optimization method that conducts search trials according to given rules.
        :param listeners: List of "observers" (used to display current information).

        :return: created process.
        """
        if parameters.number_of_parallel_points == 1:
            return Process(parameters=parameters, task=task, evolvent=evolvent,
                           search_data=search_data, method=method, listeners=listeners, calculator=calculator)
        elif parameters.async_scheme:
            return AsyncParallelProcess(parameters=parameters, task=task, evolvent=evolvent,
                                        search_data=search_data, method=method, listeners=listeners)
        else:
            return ParallelProcess(parameters=parameters, task=task, evolvent=evolvent,
                                   search_data=search_data, method=method, listeners=listeners, calculator=calculator)
