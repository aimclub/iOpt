from typing import List
import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.index_method import IndexMethod
from iOpt.method.listener import Listener
from iOpt.method.solverFactory import SolverFactory
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.parallel_process import ParallelProcess
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters
from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.method.process import Process


class Solver:
    """
    Класс Solver предназначен для выбора оптимальных (в заданной метрике) значений параметров
    сложных объектов и процессов, например, методов искусственного интеллекта и
    машинного обучения, а также – методов эвристической оптимизации.
    """

    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters = SolverParameters()
                 ):
        """
        Конструктор класса Solver

        :param problem: Постановка задачи оптимизации
        :param parameters: Параметры поиска оптимальных решений
        """

        self.problem = problem
        self.parameters = parameters

        Solver.ChackParameters(self.problem, self.parameters)

        self.__listeners: List[Listener] = []

        self.searchData = SearchData(problem)
        self.evolvent = Evolvent(problem.lowerBoundOfFloatVariables, problem.upperBoundOfFloatVariables,
                                 problem.numberOfFloatVariables)
        self.task = OptimizationTask(problem)
        self.method = SolverFactory.CreateMethod(parameters, self.task, self.evolvent, self.searchData)
        self.process = SolverFactory.CreateProcess(parameters=parameters, task=self.task, evolvent=self.evolvent,
                                   searchData=self.searchData, method=self.method, listeners=self.__listeners)

    def Solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: решение задачи оптимизации
        """
        Solver.ChackParameters(self.problem, self.parameters)
        return self.process.Solve()

    def DoGlobalIteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: число итераций глобального поиска
        """
        Solver.ChackParameters(self.problem, self.parameters)
        self.process.DoGlobalIteration(number)

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: число итераций локального поиска
        """
        Solver.ChackParameters(self.problem, self.parameters)
        self.process.DoLocalRefinement(number)

    def GetResults(self) -> Solution:
        """
        Метод позволяет получить текущую оценку решения задачи оптимизации

        :return: решение задачи оптимизации
        """
        return self.process.GetResults()

    def SaveProgress(self, fileName: str) -> None:
        """
        Сохранение процесса оптимизации в файл

        :param fileName: имя файла
        """
        self.searchData.SaveProgress(fileName=fileName)

    def LoadProgress(self, fileName: str) -> None:
        """
        Загрузка процесса оптимизации из файла

        :param fileName: имя файла
        """
        Solver.ChackParameters(self.problem, self.parameters)
        self.searchData.LoadProgress(fileName=fileName)

    def RefreshListener(self) -> None:
        """
        Метод оповещения наблюдателей о произошедшем событии
        """

        pass

    def AddListener(self, listener: Listener) -> None:
        """
        Добавления наблюдателя за процессом оптимизации

        :param listener: объект класса реализующий методы наблюдения
        """

        self.__listeners.append(listener)

    @staticmethod
    def ChackParameters(problem: Problem,
                        parameters: SolverParameters = SolverParameters()) -> None:
        """
        Проверяет параметры решателя

        :param problem: Постановка задачи оптимизации
        :param parameters: Параметры поиска оптимальных решений

        """

        if parameters.eps <= 0:
            raise Exception("search precision is incorrect, parameters.eps <= 0")
        if parameters.r <= 1:
            raise Exception("The reliability parameter should be greater 1. r>1")
        if parameters.itersLimit < 1:
            raise Exception("The number of iterations must not be negative. itersLimit>0")
        if parameters.evolventDensity < 2 or parameters.evolventDensity > 20:
            raise Exception("Evolvent density should be within [2,20]")
        if parameters.epsR < 0 or parameters.epsR >= 1:
            raise Exception("The epsilon redundancy parameter must be within [0, 1)")

        if problem.numberOfFloatVariables < 1:
            raise Exception("Must have at least one float variable")
        if problem.numberOfDisreteVariables < 0:
            raise Exception("The number of discrete parameters must not be negative")
        if problem.numberOfObjectives < 1:
            raise Exception("At least one criterion must be defined")
        if problem.numberOfConstraints < 0:
            raise Exception("The number of сonstraints must not be negative")

        if problem.floatVariableNames == [] or \
                len(problem.floatVariableNames) != problem.numberOfFloatVariables:
            raise Exception("Floaf parameter names are not defined")

        if problem.lowerBoundOfFloatVariables == [] or \
                len(problem.lowerBoundOfFloatVariables) != problem.numberOfFloatVariables:
            raise Exception("List of lower bounds for float search variables defined incorrectly")
        if problem.upperBoundOfFloatVariables == [] or \
                len(problem.upperBoundOfFloatVariables) != problem.numberOfFloatVariables:
            raise Exception("List of upper bounds for float search variables defined incorrectly")

        for lowerBound, upperBound in zip(problem.lowerBoundOfFloatVariables, problem.upperBoundOfFloatVariables):
            if lowerBound >= upperBound:
                raise Exception("For floating point search variables, "
                                "the upper search bound must be greater than the lower.")

        if problem.numberOfDisreteVariables > 0:
            if problem.discreteVariableNames == [] or \
                    len(problem.discreteVariableNames) != problem.numberOfDisreteVariables:
                raise Exception("Discrete parameter names are not defined")

            for discreteValues in problem.discreteVariableValues:
                if len(discreteValues) < 1:
                    raise Exception("Discrete variable values not defined")

        if parameters.startPoint != []:
            raise Exception("At the moment, the starting point is not used")
