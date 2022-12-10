from typing import List
import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.optim_task import OptimizationTask
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
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: решение задачи оптимизации
        """
        return self.process.Solve()

    def DoGlobalIteration(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций глобального поиска

        :param number: число итераций глобального поиска
        """
        self.process.DoGlobalIteration(number)

    def DoLocalRefinement(self, number: int = 1):
        """
        Метод позволяет выполнить несколько итераций локального поиска

        :param number: число итераций локального поиска
        """
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
