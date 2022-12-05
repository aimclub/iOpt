from typing import List

from iOpt.problem import Problem
from iOpt.trial import Point, Trial


class Solution:
    def __init__(self,
                 problem: Problem,
                 bestTrials: List[Trial] = [Trial(Point([], []), [])],

                 numberOfGlobalTrials: int = 0,
                 numberOfLocalTrials: int = 0,
                 solvingTime: float = 0.0,
                 solutionAccuracy: float = 0.0
                 ):
        self.problem = problem
        self.bestTrials = bestTrials

        self.numberOfGlobalTrials = numberOfGlobalTrials
        self.numberOfLocalTrials = numberOfLocalTrials
        self.solvingTime = solvingTime
        self.solutionAccuracy = solutionAccuracy
