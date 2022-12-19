from iOpt.trial import Trial
from iOpt.problem import Problem
import numpy as np


class Solution:
    """
    Класс описания решения задачи оптимизации
    """
    def __init__(self,
                 problem: Problem,
                 bestTrials: np.ndarray(shape=(1), dtype=Trial) = [Trial([], [])],

                 numberOfGlobalTrials: int = 0,
                 numberOfLocalTrials: int = 0,
                 solvingTime: np.double = 0.0,
                 solutionAccuracy: np.double = 0.0
                 ):
        """
        Конструктор класса

        :param problem: Задача оптимизации
        :param bestTrials: Решение задачи оптимизации
        :param numberOfGlobalTrials: Число выполненных глобальных итераций поиска
        :param numberOfLocalTrials: Число выполненных локальных итераций поиска
        :param solvingTime: Время решения задачи
        :param solutionAccuracy: Точность найденного решения
        """

        self.problem = problem
        self.bestTrials = bestTrials

        self.numberOfGlobalTrials = numberOfGlobalTrials
        self.numberOfLocalTrials = numberOfLocalTrials
        self.solvingTime = solvingTime
        self.solutionAccuracy = solutionAccuracy
