from typing import List

from iOpt.method.listener import Listener
from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.search_data import SearchData
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.method import Method
from iOpt.problem import Problem
from iOpt.solver import SolverParameters
from iOpt.solution import Solution


class Process:
    __listeners: List[Listener] = []
    __first_iteration = False

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData,
                 method: Method
                 ):
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
        # if self.__first_iteration is False:
        #     self.method.FirstIteration()
        #     self.__first_iteration = True
        while self.method.CheckStopCondition() is False:
            self.DoGlobalIteration()
        return self.GetResults()

    def DoGlobalIteration(self, number: int = 1):
        """
        :param number: The number of iterations of the global search
        """
        for _ in range(number):
            if self.__first_iteration is False:
                self.method.FirstIteration()
                self.__first_iteration = True
            else:
                newoldpont = self.method.CalculateIterationPoint()
                self.method.CalculateFunctionals(newoldpont[0])
                self.method.UpdateOptimum(newoldpont[0])
                self.method.RenewSearchData(newoldpont[0], newoldpont[1])
                self.method.FinalizeIteration()

    def DoLocalRefinement(self, number: int = 1):
        """
        :param number: The number of iterations of the local search
        """
        pass

    def GetResults(self) -> Solution:
        """
        :return: Return current solution for the optimization problem
        """
        # ДА, ЭТО КОСТЫЛЬ. т.к. solution хранит trial
        self.searchData.solution.bestTrials[0] = self.method.GetOptimumEstimation()
        return self.searchData.solution

    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        self.__listeners.append(listener)
