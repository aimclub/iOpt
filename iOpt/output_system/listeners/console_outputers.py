from typing import List

from iOpt.method.listener import Listener
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.method.method import Method
from iOpt.output_system.outputers.console_outputer import ConsoleOutputer

import numpy as np


class ConsoleOutputListener(Listener):
    """
    The ConsoleOutputListener class is an event listener. It contains handler methods that produce console output as a
      console output as a reaction to the event
    """

    def __init__(self, mode='full', iters=100):
        """
        Constructor of the ConsoleOutputListener class

        :param mode: The console output mode to be used. Possible modes: 'full', 'custom' and 'result'.
           The 'full' mode performs full output of the search information obtained by the method to the console during the optimization process.
           information received by the method. The 'custom' mode outputs the current best point with a specified frequency. 'result' mode
           mode outputs to the console only the final result of the optimization process.
        :param iters: Frequency of output to the console. Used in conjunction with the 'custom' output mode.
        """
        self.__outputer: ConsoleOutputer = None
        self.mode = mode
        self.iters = iters

    def before_method_start(self, method: Method):
        self.__outputer = ConsoleOutputer(method.task.problem, method.parameters)
        self.__outputer.print_init_info()

    def on_end_iteration(self, curr_points: List[SearchDataItem], curr_solution: Solution):
        if self.mode == 'full':
            self.__outputer.print_iter_point_info(curr_points)
        elif self.mode == 'custom':
            self.__outputer.print_best_point_info(curr_solution, self.iters)
        elif self.mode == 'result':
            pass

    def on_method_stop(self, search_data: SearchData, solution: Solution, status: bool):
        self.__outputer.print_final_result_info(solution, status)
        if self.mode == 'full' and solution.best_trials.size > 1:
            self.__outputer.print_pareto_set_info(solution)
