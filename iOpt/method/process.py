import sys
from datetime import datetime
from typing import List

import traceback

import scipy
from scipy.optimize import Bounds

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.local_optimizer import LocalOptimize
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import FunctionValue, FunctionType
from iOpt.trial import Point


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

        :param parameters: Параметры решения задачи оптимизации.
        :param task: Обёртка решаемой задачи.
        :param evolvent: Развертка Пеано-Гильберта, отображающая отрезок [0,1] на многомерную область D.
        :param searchData: Структура данных для хранения накопленной поисковой информации.
        :param method: Метод оптимизации, проводящий поисковые испытания по заданным правилам.
        :param listeners: Список "наблюдателей" (используется для вывода текущей информации).
        """
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
        self.method = method
        self._listeners = listeners
        self._first_iteration = True

    def Solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: Текущая оценка решения задачи оптимизации
        """

        startTime = datetime.now()

        try:
            while not self.method.CheckStopCondition():
                self.DoGlobalIteration()

        except Exception:
            print('Exception was thrown')
            print(traceback.format_exc())

        if self.parameters.refineSolution:
            self.DoLocalRefinement(self.parameters.localMethodIterationCount)

        result = self.GetResults()
        result.solvingTime = (datetime.now() - startTime).total_seconds()

        for listener in self._listeners:
            status = self.method.CheckStopCondition()
            listener.OnMethodStop(self.searchData, self.GetResults(), status)

        return result

    def DoGlobalIteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: Количество итераций глобального поиска
        """
        number_ = number
        doneTrials = []
        if self._first_iteration is True:
            for listener in self._listeners:
                listener.BeforeMethodStart(self.method)
            doneTrials = self.method.FirstIteration()
            self._first_iteration = False
            number = number - 1

        for _ in range(number):
            newpoint, oldpoint = self.method.CalculateIterationPoint()
            self.method.CalculateFunctionals(newpoint)
            self.method.UpdateOptimum(newpoint)
            self.method.RenewSearchData(newpoint, oldpoint)
            self.method.FinalizeIteration()
            doneTrials = self.searchData.GetLastItems(self.parameters.numberOfParallelPoints * number_)

        for listener in self._listeners:
            listener.OnEndIteration(doneTrials, self.GetResults())

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: Количество итераций локального поиска
        """
        try:
            local_method_iteration_count = number
            if number == -1:
                local_method_iteration_count = int(self.parameters.localMethodIterationCount)

            result = self.GetResults()
            # start_point = result.bestTrials[0].point.floatVariables

            local_solution = LocalOptimize(self.task,
                                           method="Hooke-Jeeves", start_point=result.bestTrials[0].point,
                                           max_calcs=local_method_iteration_count,
                                           args={"eps": self.parameters.eps / 100, "step_mult": 2,
                                                 "max_iter": local_method_iteration_count}
                                           )
            # scipy.optimize.minimize(self.problemCalculate, x0=start_point, method='Nelder-Mead',
            #                        options={'maxiter': local_method_iteration_count})
            # local_solution = LocalOptimize(LocalTaskWrapper(self.task, result.bestTrials[0].point.discreteVariables),
            #                               method="Nelder-Mead", start_point=start_point,
            #                               args={"options": {'maxiter': local_method_iteration_count}})

            if local_method_iteration_count > 0:
                result.bestTrials[0].point.floatVariables = local_solution["x"]

                point: SearchDataItem = SearchDataItem(result.bestTrials[0].point,
                                                       self.evolvent.GetInverseImage(
                                                           result.bestTrials[0].point.floatVariables),
                                                       functionValues=[FunctionValue()] *
                                                                      (self.task.problem.numberOfConstraints +
                                                                       self.task.problem.numberOfObjectives)
                                                       )

                number_of_constraints = self.task.problem.numberOfConstraints
                for i in range(number_of_constraints):
                    point.functionValues[i] = FunctionValue(FunctionType.CONSTRAINT, i)
                    point.functionValues[i] = self.task.problem.Calculate(point.point, point.functionValues[i])
                    point.SetZ(point.functionValues[i].value)
                    point.SetIndex(i)
                    if point.GetZ() > 0:
                        break
                point.functionValues[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV,
                                                                            number_of_constraints)
                point.functionValues[number_of_constraints] = \
                    self.task.problem.Calculate(point.point, point.functionValues[number_of_constraints])
                point.SetZ(point.functionValues[number_of_constraints].value)
                point.SetIndex(number_of_constraints)

                result.bestTrials[0].functionValues = point.functionValues

            result.numberOfLocalTrials = local_solution["fev"]
        except Exception:
            print("Local Refinement is not possible")

    def GetResults(self) -> Solution:
        """
        Метод возвращает лучшее найденное решение задачи оптимизации

        :return: Решение задачи оптимизации
        """
        return self.searchData.solution

    def SaveProgress(self, fileName: str) -> None:
        """
        Сохранение процесса оптимизации из файла

        :param fileName: имя файла
        """
        self.searchData.SaveProgress(fileName=fileName)

    def LoadProgress(self, fileName: str) -> None:
        """
        Загрузка процесса оптимизации из файла

        :param fileName: имя файла
        """
        self.searchData.LoadProgress(fileName=fileName)
        self.method.iterationsCount = self.searchData.GetCount() - 2

        for ditem in self.searchData:
            if ditem.GetIndex() >= 0:
                self.method.UpdateOptimum(ditem)

        self.method.RecalcM()
        self.method.RecalcAllCharacteristics()
        self._first_iteration = False

        for listener in self._listeners:
            listener.BeforeMethodStart(self.method)

    '''
    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        #self.__listeners.append(listener)
        pass
    '''
