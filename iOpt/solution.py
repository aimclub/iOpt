from iOpt.trial import Trial
from iOpt.problem import Problem
import numpy as np


class Solution:
    """
    Класс описания решения задачи оптимизации
    """
    def __init__(self,
                 problem: Problem,
                 best_trials: np.ndarray(shape=(1), dtype=Trial) = [Trial([], [])],

                 number_of_global_trials: int = 1,
                 number_of_local_trials: int = 0,
                 solving_time: np.double = 0.0,
                 solution_accuracy: np.double = 0.0
                 ):
        """
        Конструктор класса

        :param problem: Задача оптимизации
        :param best_trials: Решение задачи оптимизации
        :param number_of_global_trials: Число выполненных глобальных итераций поиска
        :param number_of_local_trials: Число выполненных локальных итераций поиска
        :param solving_time: Время решения задачи
        :param solution_accuracy: Точность найденного решения
        """

        self.problem = problem
        self.best_trials = best_trials

        self.number_of_global_trials = number_of_global_trials
        self.number_of_local_trials = number_of_local_trials
        self.solving_time = solving_time
        self.solution_accuracy = solution_accuracy
