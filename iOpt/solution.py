from iOpt.trial import Trial
from iOpt.problem import Problem
import numpy as np


class Solution:
    """
    Class of description of the solution to the optimization problem
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
        Class constructor

        :param problem: Optimization problem.
        :param best_trials: Solution the optimization problem.
        :param number_of_global_trials: Number of global search iterations performed.
        :param number_of_local_trials: Number of local search iterations performed.
        :param solving_time: Problem solution time.
        :param solution_accuracy: Accuracy of the solution found.
        """

        self.problem = problem
        self.best_trials = best_trials

        self.number_of_global_trials = number_of_global_trials
        self.number_of_local_trials = number_of_local_trials
        self.solving_time = solving_time
        self.solution_accuracy = solution_accuracy
