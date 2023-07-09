from typing import List

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.optim_task import OptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.method.solverFactory import SolverFactory
from iOpt.problem import Problem
from iOpt.routine.timeout import timeout
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters


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

        Solver.CheckParameters(self.problem, self.parameters)

        self.__listeners: List[Listener] = []

        self.searchData = SearchData(problem)
        self.evolvent = Evolvent(problem.lowerBoundOfFloatVariables, problem.upperBoundOfFloatVariables,
                                 problem.numberOfFloatVariables)
        self.task = OptimizationTask(problem)
        self.method = SolverFactory.CreateMethod(parameters, self.task, self.evolvent, self.searchData)
        self.process = SolverFactory.CreateProcess(parameters=parameters, task=self.task, evolvent=self.evolvent,
                                                   searchData=self.searchData, method=self.method,
                                                   listeners=self.__listeners)

    def Solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: решение задачи оптимизации
        """
        Solver.CheckParameters(self.problem, self.parameters)
        sol: Solution = None
        if self.parameters.timeout < 0:
            sol = self.process.Solve()
        else:
            solv_with_timeout = timeout(seconds=self.parameters.timeout * 60)(self.process.Solve)
            try:
                solv_with_timeout()
            except Exception as exc:
                print(exc)
                sol = self.GetResults()
                sol.solvingTime = self.parameters.timeout * 60
                self.method.recalcR = True
                self.method.recalcM = True
                status = self.method.CheckStopCondition()
                for listener in self.__listeners:
                    listener.OnMethodStop(self.searchData, self.GetResults(), status)
        return sol

    def DoGlobalIteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: число итераций глобального поиска
        """
        Solver.CheckParameters(self.problem, self.parameters)
        self.process.DoGlobalIteration(number)

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: число итераций локального поиска
        """
        Solver.CheckParameters(self.problem, self.parameters)
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
        self.process.SaveProgress(fileName=fileName)

    def LoadProgress(self, fileName: str) -> None:
        """
        Загрузка процесса оптимизации из файла

        :param fileName: имя файла
        """
        Solver.CheckParameters(self.problem, self.parameters)
        self.process.LoadProgress(fileName=fileName)

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
    def CheckParameters(problem: Problem,
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
        if problem.numberOfDiscreteVariables < 0:
            raise Exception("The number of discrete parameters must not be negative")
        if problem.numberOfObjectives < 1:
            raise Exception("At least one criterion must be defined")
        if problem.numberOfConstraints < 0:
            raise Exception("The number of сonstraints must not be negative")

        if len(problem.floatVariableNames) != problem.numberOfFloatVariables:
            raise Exception("Floaf parameter names are not defined")

        if len(problem.lowerBoundOfFloatVariables) != problem.numberOfFloatVariables:
            raise Exception("List of lower bounds for float search variables defined incorrectly")
        if len(problem.upperBoundOfFloatVariables) != problem.numberOfFloatVariables:
            raise Exception("List of upper bounds for float search variables defined incorrectly")

        for lowerBound, upperBound in zip(problem.lowerBoundOfFloatVariables, problem.upperBoundOfFloatVariables):
            if lowerBound >= upperBound:
                raise Exception("For floating point search variables, "
                                "the upper search bound must be greater than the lower.")

        if problem.numberOfDiscreteVariables > 0:
            if len(problem.discreteVariableNames) != problem.numberOfDiscreteVariables:
                raise Exception("Discrete parameter names are not defined")

            for discreteValues in problem.discreteVariableValues:
                if len(discreteValues) < 1:
                    raise Exception("Discrete variable values not defined")

        if parameters.startPoint:
            if len(parameters.startPoint.floatVariables) != problem.numberOfFloatVariables:
                raise Exception("Incorrect start point size")
            if parameters.startPoint.discreteVariables:
                if len(parameters.startPoint.discreteVariables) != problem.numberOfDiscreteVariables:
                    raise Exception("Incorrect start point discrete variables")
            for lowerBound, upperBound, y in zip(problem.lowerBoundOfFloatVariables, problem.upperBoundOfFloatVariables,
                                                 parameters.startPoint.floatVariables):
                if y < lowerBound or y > upperBound:
                    raise Exception("Incorrect start point coordinate")

