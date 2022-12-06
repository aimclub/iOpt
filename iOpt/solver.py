from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.process import Process
from iOpt.method.search_data import SearchData
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


class Solver:
    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters = SolverParameters()
                 ):
        """
        :param problem: Optimization problem
        :param parameters: Parameters for solving the problem
        """
        self.problem = problem
        self.parameters = parameters

        self.__listeners: List[Listener] = []

        self.searchData = SearchData(problem)
        self.evolvent = Evolvent(problem.lowerBoundOfFloatVariables, problem.upperBoundOfFloatVariables,
                                 problem.numberOfFloatVariables)
        self.task = OptimizationTask(problem)
        self.method = Method(parameters, self.task, self.evolvent, self.searchData)
        self.process = Process(parameters=parameters, task=self.task, evolvent=self.evolvent,
                               searchData=self.searchData, method=self.method, listeners=self.__listeners)

    def Solve(self) -> Solution:
        """
        Retrieve a solution with check of the stop conditions
        :return: Solution for the optimization problem
        """
        return self.process.Solve()

    def DoGlobalIteration(self, number: int = 1) -> None:
        """
        :param number: The number of iterations of the global search
        """
        self.process.DoGlobalIteration(number)

    def DoLocalRefinement(self, number: int = 1) -> None:
        """
        :param number: The number of iterations of the local search
        """
        self.process.DoLocalRefinement(number)

    def GetResults(self) -> Solution:
        """
        :return: Return current solution for the optimization problem
        """
        return self.process.GetResults()

    def SaveProgress(self, fileName: str) -> None:
        """
        :return:
        """
        self.searchData.SaveProgress(fileName=fileName)

    def LoadProgress(self, fileName: str) -> None:
        """
        :return:
        """
        self.searchData.LoadProgress(fileName=fileName)

    def RefreshListener(self) -> None:
        pass

    def AddListener(self, listener: Listener) -> None:
        self.__listeners.append(listener)
