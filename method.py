from __future__ import annotations

from evolvent import Evolvent
from search_data import SearchData
from search_data import SearchDataItem
from optim_task import OptimizationTask
from problem import Problem
from solver import SolverParameters


class Method:
    stop: bool = False

    def __init__(self,
                 problem: Problem,
                 parameters: SolverParameters,
                 task: OptimizationTask,
                 evolvent: Evolvent,
                 searchData: SearchData
                ):
        self.problem = problem
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData

    def FirstIteration(self):
        pass

    def CheckStopCondition(self):
        pass

    def RecalcAllCharacteristics(self):
        pass

    def CalculateIterationPoint(self) -> SearchDataItem:
        pass

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        pass

    def RenewSearchData(self, point: SearchDataItem):
        pass

    def UpdateOptimum(self, point: SearchDataItem):
        pass

    def FinalizeIteration(self):
        pass

