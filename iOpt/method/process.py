from typing import List

from iOpt.method.listener import Listener
from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.search_data import SearchData
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.method import Method
from iOpt.problem import Problem
from iOpt.solver_parametrs import SolverParameters
from iOpt.solution import Solution
from iOpt.trial import Point
from iOpt.trial import FunctionValue

from datetime import datetime
import time

import scipy
from scipy.optimize import Bounds


class Process:
    """
    Класс Process скрывает внутреннюю имплементацию класса Solver.
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData,
                 method: Method,
                 listeners: List[Listener]
                 ):
        """
        Конструктор класса Process

        :param parameters: Параметры поиска оптимальных решений
        :param task: Обёртка решаемой задачи.
        :param evolvent: Развёртка.
        :param searchData: Хранилище данных.
        :param method: Метод, проводящий испытания по заданным правилам.
        :param listeners: Список наблюдателей.
        """
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
        self.method = method
        self.__listeners = listeners
        self.__first_iteration = True

    def Solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно параметрам оптимизации,
        полученным при создании класса Solver.

        :return: Решение задачи оптимизации
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

        if self.parameters.refineSolution:
            self.DoLocalRefinement(-1)

        for listener in self.__listeners:
            status = self.method.CheckStopCondition()
            listener.OnMethodStop(self.searchData, self.GetResults(), status)

        result = self.GetResults()
        result.solvingTime = datetime.now() - startTime
        return result

    def DoGlobalIteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: Количество итераций глобального поиска
        """
        savedNewPoints = []
        for _ in range(number):
            if self.__first_iteration is True:
                for listener in self.__listeners:
                    listener.BeforeMethodStart(self.method)
                self.method.FirstIteration()
                savedNewPoints.append(self.searchData.GetLastItem())
                self.__first_iteration = False
            else:
                newpoint, oldpoint = self.method.CalculateIterationPoint()
                savedNewPoints.append(newpoint)
                self.method.CalculateFunctionals(newpoint)
                self.method.UpdateOptimum(newpoint)
                self.method.RenewSearchData(newpoint, oldpoint)
                self.method.FinalizeIteration()

        for listener in self.__listeners:
            listener.OnEndIteration(savedNewPoints)

    def problemCalculate(self, y):
        point = Point(y, [])
        functionValue = FunctionValue()
        functionValue = self.task.problem.Calculate(point, functionValue)
        return functionValue.value

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: Количество итераций локального поиска
        """
        self.localMethodIterationCount = number
        if number == -1:
            self.localMethodIterationCount = self.parameters.itersLimit * 0.9

        result = self.GetResults()
        startPoint = result.bestTrials[0].point.floatVariables

        bounds = Bounds(self.task.problem.lowerBoundOfFloatVariables, self.task.problem.upperBoundOfFloatVariables)

        # nelder_mead = scipy.optimize.minimize(self.problemCalculate, x0 = startPoint, method='Nelder-Mead',
        # options={'maxiter':self.localMethodIterationCount}, bounds=bounds)

        nelder_mead = scipy.optimize.minimize(self.problemCalculate, x0=startPoint, method='Nelder-Mead',
                                              options={'maxiter': self.localMethodIterationCount})

        result.bestTrials[0].point.floatVariables = nelder_mead.x
        result.bestTrials[0].functionValues[0].value = self.problemCalculate(result.bestTrials[0].point.floatVariables)

        result.numberOfLocalTrials = nelder_mead.nfev

    def GetResults(self) -> Solution:
        """
        Метод позволяет получить достигнутое решение задачи оптимизации

        :return: Решение задачи оптимизации
        """
        # ДА, ЭТО КОСТЫЛЬ. т.к. solution хранит trial
        # self.searchData.solution.bestTrials[0] = self.method.GetOptimumEstimation()
        return self.searchData.solution

    '''
    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        #self.__listeners.append(listener)
        pass
    '''
