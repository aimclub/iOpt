from datetime import datetime
from typing import List

import scipy
from scipy.optimize import Bounds

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, Point


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
        startTime = datetime.now()

        try:
            while not self.method.CheckStopCondition():
                self.DoGlobalIteration()
                # print(self.method.min_delta, self.method.parameters.eps)
            # print(self.method.min_delta, self.method.parameters.eps)
            # print(self.method.CheckStopCondition())
        except BaseException:
            print('Exception was thrown')

        for listener in self.__listeners:
            status = self.method.CheckStopCondition()
            listener.OnMethodStop(self.searchData, self.GetResults(), status)
        if self.parameters.refineSolution:
            self.DoLocalRefinement(-1)

        result = self.GetResults()
        result.solvingTime = (datetime.now() - startTime).total_seconds()

        return result

    def DoGlobalIteration(self, number: int = 1) -> None:
        """
        :param number: The number of iterations of the global search
        """
        savedNewPoints: List[SearchDataItem] = []
        for _ in range(number):
            if not self.__first_iteration:
                for listener in self.__listeners:
                    listener.BeforeMethodStart(self.method)
                self.method.FirstIteration()
                savedNewPoints.append(self.searchData.GetLastItem())
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

    def problemCalculate(self, y: List[float]) -> float:
        point = Point(y, [])
        functionValue = FunctionValue()
        functionValue = self.task.problem.Calculate(point, functionValue)
        return functionValue.value

    def DoLocalRefinement(self, number: int = 1) -> None:
        """
        :param number: The number of iterations of the local search
        """
        self.localMethodIterationCount = float(number)
        if number == -1:
            self.localMethodIterationCount = self.parameters.itersLimit * 0.9

        result = self.GetResults()
        startPoint = result.bestTrials[0].point.floatVariables

        bounds = Bounds(self.task.problem.lowerBoundOfFloatVariables, self.task.problem.upperBoundOfFloatVariables)

        nelder_mead = scipy.optimize.minimize(
            self.problemCalculate,
            x0=startPoint,
            method='Nelder-Mead',
            options={
                'maxiter': self.localMethodIterationCount},
            bounds=bounds)

        result.bestTrials[0].point.floatVariables = nelder_mead.x
        result.bestTrials[0].functionValues[0].value = self.problemCalculate(result.bestTrials[0].point.floatVariables)

        result.numberOfLocalTrials = nelder_mead.nfev

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
