from typing import List

from listener import Listener
from evolvent import Evolvent
from search_data import SearchData
from optim_task import OptimizationTask
from method import Method
from problem import Problem
from solver import SolverParameters
from solver import Solution


class Process:
    __listeners: List[Listener] = []

    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData,
                 method: Method
                ):
        self.problem = problem
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
        self.method = method

    def Solve(self) -> Solution:
        """
        Retrieve a solution with check of the stop conditions
        :return: Solution for the optimization problem
        """

    def DoGlobalIteration(self, number: int = 1):
        """
        :param number: The number of iterations of the global search
        """

    def DoLocalRefinement(self, number: int = 1):
        """
        :param number: The number of iterations of the local search
        """

    def GetResults(self) -> Solution:
        """
        :return: Return current solution for the optimization problem
        """

    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        self.__listeners.append(listener)
