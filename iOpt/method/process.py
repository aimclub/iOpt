from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


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
        try:
            while not self.method.CheckStopCondition():
                self.DoGlobalIteration()
                # print(self.method.min_delta, self.method.parameters.eps)
            # print(self.method.min_delta, self.method.parameters.eps)
            print(self.method.CheckStopCondition())
        except BaseException:
            print('Exception was thrown')
        for listener in self.__listeners:
            listener.OnMethodStop(self.searchData)
        return self.GetResults()

    def DoGlobalIteration(self, number: int = 1) -> None:
        """
        :param number: The number of iterations of the global search
        """
        for _ in range(number):
            if self.__first_iteration is False:
                for listener in self.__listeners:
                    listener.BeforeMethodStart(self.searchData)
                self.method.FirstIteration()
                self.__first_iteration = True
            else:
                newpoint, oldpoint = self.method.CalculateIterationPoint()
                self.method.CalculateFunctionals(newpoint)
                self.method.UpdateOptimum(newpoint)
                self.method.RenewSearchData(newpoint, oldpoint)
                self.method.FinalizeIteration()
        for listener in self.__listeners:
            listener.OnEndIteration(self.searchData)

    def DoLocalRefinement(self, number: int = 1) -> None:
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

    def RefreshListener(self) -> None:
        pass

    def AddListener(self, listener: Listener) -> None:
        self.__listeners.append(listener)
