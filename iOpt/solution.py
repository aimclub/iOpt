import numpy as np

from iOpt.problem import Problem
from iOpt.trial import Trial


class Solution:
    def __init__(self,
                 problem: Problem,
                 bestTrials: np.ndarray(shape=(1), dtype=Trial) = [Trial([], [])],

                 numberOfGlobalTrials: int = 0,
                 numberOfLocalTrials: int = 0,
                 solvingTime: np.double = 0.0,
                 solutionAccuracy: np.double = 0.0
                 ):
        self.problem = problem
        self.bestTrials = bestTrials

        self.numberOfGlobalTrials = numberOfGlobalTrials
        self.numberOfLocalTrials = numberOfLocalTrials
        self.solvingTime = solvingTime
        self.solutionAccuracy = solutionAccuracy
