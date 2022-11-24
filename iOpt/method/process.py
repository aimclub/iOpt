from typing import List

from iOpt.method.listener import Listener
from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.search_data import SearchData
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.method import Method
from iOpt.problem import Problem
from iOpt.solver_parametrs import SolverParameters
from iOpt.solution import Solution


class Process:
    __first_iteration = False

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData,
                 method: Method,
                 listeners: List[Listener]
                 ):
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
        self.method = method
        self.__listeners = listeners

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
            #print(self.method.CheckStopCondition())
        except:
            print('Exception was thrown')

        for listener in self.__listeners:
            status = self.method.CheckStopCondition()
            listener.OnMethodStop(self.searchData, self.GetResults(), status)
        return self.GetResults()

    def DoGlobalIteration(self, number: int = 1):
        """
        :param number: The number of iterations of the global search
        """
        savedNewPoints = []
        for _ in range(number):
            if self.__first_iteration is False:
                for listener in self.__listeners:
                    listener.BeforeMethodStart(self.method)
                self.method.FirstIteration()    
                # костыль
                i = -1
                for item in self.searchData:
                    i = i + 1
                    if i == 1:
                        savedNewPoints.append(item)
                        break
                # конец костыля
                self.__first_iteration = True
            else:
                newpoint, oldpoint = self.method.CalculateIterationPoint()
                savedNewPoints.append(newpoint)
                self.method.CalculateFunctionals(newpoint)
                self.method.UpdateOptimum(newpoint)
                self.method.RenewSearchData(newpoint, oldpoint)
                self.method.FinalizeIteration()

        for listener in self.__listeners:
            listener.OnEndIteration(savedNewPoints)


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
    '''
    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        #self.__listeners.append(listener)
        pass
    '''
