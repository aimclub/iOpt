from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.listener import Listener
from iOpt.method.method import Method

from iOpt.method.optim_task import OptimizationTask
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solver_parametrs import SolverParameters


class ParallelProcess(Process):
    """
    The ParallelProcess class implements parallelization at the level of threads (python processes)
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener],
                 calculator: Calculator = None
                 ):
        """
        Constructor of the ParallelProcess class

        :param parameters: Parameters of the solution to the optimization problem.
        :param task: The wrapper of the problem to be solved.
        :param evolvent: Peano-Hilbert evolvent mapping the segment [0,1] to the multidimensional region D.
        :param search_data: A data structure for storing accumulated search information.
        :param method: An optimization method that performs search trials according to given rules.
        :param listeners: List of "observers" (used to display current information).
        """
        super(ParallelProcess, self).__init__(parameters, task, evolvent, search_data, method, listeners, calculator)

    def do_global_iteration(self, number: int = 1):
        """
        Perform several iterations of the global search

        :param number: Number of iterations of global search.
        """
        number_ = number
        done_trials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.before_method_start(self.method)
            done_trials = self.method.first_iteration()
            self._first_iteration = False
            number -= 1

        for _ in range(number):
            list_newpoint: list[SearchDataItem] = []
            list_oldpoint: list[SearchDataItem] = []

            for _ in range(self.parameters.number_of_parallel_points):
                newpoint, oldpoint = self.method.calculate_iteration_point()
                list_newpoint.append(newpoint)
                list_oldpoint.append(oldpoint)
            self.calculator.calculate_functionals_for_items(list_newpoint)

            for newpoint, oldpoint in zip(list_newpoint, list_oldpoint):
                self.method.update_optimum(newpoint)
                self.method.renew_search_data(newpoint, oldpoint)
                self.method.finalize_iteration()
                done_trials = self.search_data.get_last_items(self.parameters.number_of_parallel_points * number_)

        for listener in self._listeners:
            listener.on_end_iteration(done_trials, self.get_results())
