from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.async_parallel_process import AsyncParallelProcess
from iOpt.method.index_method import IndexMethod
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.mixed_integer_method import MixedIntegerMethod
from iOpt.method.optim_task import OptimizationTask
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
    def create_method(parameters: SolverParameters,
                      task: OptimizationTask,
                      evolvent: Evolvent,
                      search_data: SearchData) -> Method:
        """
        Create a suitable solution method class based on the given parameters

        :param parameters: parameters of the solution of the optimization problem.
        :param task: the wrapper of the problem to be solved.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: data structure for storing accumulated search information.

        :return: created method
        """

        if task.problem.number_of_discrete_variables > 0:
            return MixedIntegerMethod(parameters, task, evolvent, search_data)
        elif task.problem.number_of_constraints > 0:
            return IndexMethod(parameters, task, evolvent, search_data)
        else:
            return Method(parameters, task, evolvent, search_data)

    @staticmethod
    def create_process(parameters: SolverParameters,
                       task: OptimizationTask,
                       evolvent: Evolvent,
                       search_data: SearchData,
                       method: Method,
                       listeners: List[Listener]) -> Process:
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
                           search_data=search_data, method=method, listeners=listeners)
        elif parameters.async_scheme:
            return AsyncParallelProcess(parameters=parameters, task=task, evolvent=evolvent,
                                        search_data=search_data, method=method, listeners=listeners)
        else:
            return ParallelProcess(parameters=parameters, task=task, evolvent=evolvent,
                                   search_data=search_data, method=method, listeners=listeners)

