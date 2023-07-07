import sys
from datetime import datetime
from typing import List

import traceback

import scipy
from scipy.optimize import Bounds

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
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
        self.search_data = searchData
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

        if self.parameters.refine_solution:
            self.DoLocalRefinement(self.parameters.local_method_iteration_count)

        result = self.GetResults()
        result.solving_time = (datetime.now() - startTime).total_seconds()

        for listener in self._listeners:
            status = self.method.CheckStopCondition()
            listener.OnMethodStop(self.search_data, self.GetResults(), status)

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
            doneTrials = self.search_data.GetLastItems(self.parameters.number_of_parallel_points * number_)

        for listener in self._listeners:
            listener.OnEndIteration(doneTrials, self.GetResults())

    def problemCalculate(self, y):
        result = self.GetResults()
        point = Point(y, result.best_trials[0].point.discrete_variables)
        functionValue = FunctionValue(FunctionType.OBJECTIV)

        for i in range(self.task.problem.dimension):
            if (y[i] < self.task.problem.lower_bound_of_float_variables[i]) \
                    or (y[i] > self.task.problem.upper_bound_of_float_variables[i]):
                functionValue.value = sys.float_info.max
                return functionValue.value

        try:
            for i in range(self.task.problem.number_of_constraints):
                functionConstraintValue = FunctionValue(FunctionType.CONSTRAINT, i)
                functionConstraintValue = self.task.problem.calculate(point, functionConstraintValue)
                if functionConstraintValue.value > 0:
                    functionValue.value = sys.float_info.max
                    return functionValue.value

            functionValue = self.task.problem.calculate(point, functionValue)
        except Exception:
            functionValue.value = sys.float_info.max

        return functionValue.value

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: Количество итераций локального поиска
        """
        try:
            local_method_iteration_count = number
            if number == -1:
                local_method_iteration_count = self.parameters.local_method_iteration_count

            result = self.GetResults()
            start_point = result.best_trials[0].point.float_variables

            nelder_mead = scipy.optimize.minimize(self.problemCalculate, x0=start_point, method='Nelder-Mead',
                                                  options={'maxiter': local_method_iteration_count})

            if local_method_iteration_count > 0:
                result.best_trials[0].point.float_variables = nelder_mead.x

                point: SearchDataItem = SearchDataItem(result.best_trials[0].point,
                                                       self.evolvent.GetInverseImage(
                                                           result.best_trials[0].point.float_variables),
                                                       function_values=[FunctionValue()] *
                                                                       (self.task.problem.number_of_constraints +
                                                                       self.task.problem.number_of_objectives)
                                                       )

                number_of_constraints = self.task.problem.number_of_constraints
                for i in range(number_of_constraints):
                    point.function_values[i] = FunctionValue(FunctionType.CONSTRAINT, i)
                    point.function_values[i] = self.task.problem.calculate(point.point, point.function_values[i])
                    point.SetZ(point.function_values[i].value)
                    point.SetIndex(i)
                    if point.GetZ() > 0:
                        break
                point.function_values[number_of_constraints] = FunctionValue(FunctionType.OBJECTIV,
                                                                            number_of_constraints)
                point.function_values[number_of_constraints] = \
                    self.task.problem.calculate(point.point, point.function_values[number_of_constraints])
                point.SetZ(point.function_values[number_of_constraints].value)
                point.SetIndex(number_of_constraints)

                result.best_trials[0].function_values = point.function_values

            result.number_of_local_trials = nelder_mead.nfev
        except Exception:
            print("Local Refinement is not possible")

    def GetResults(self) -> Solution:
        """
        Метод возвращает лучшее найденное решение задачи оптимизации

        :return: Решение задачи оптимизации
        """
        # ДА, ЭТО КОСТЫЛЬ. т.к. solution хранит trial
        # self.search_data.solution.best_trials[0] = self.method.GetOptimumEstimation()
        return self.search_data.solution

    def SaveProgress(self, fileName: str) -> None:
        """
        Сохранение процесса оптимизации из файла

        :param fileName: имя файла
        """
        self.search_data.SaveProgress(fileName=fileName)

    def LoadProgress(self, fileName: str) -> None:
        """
        Загрузка процесса оптимизации из файла

        :param fileName: имя файла
        """
        self.search_data.LoadProgress(fileName=fileName)
        self.method.iterationsCount = self.search_data.GetCount() - 2

        for ditem in self.search_data:
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
