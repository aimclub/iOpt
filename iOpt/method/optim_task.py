from __future__ import annotations
import numpy as np
from enum import Enum

from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem


class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class OptimizationTask:
    def __init__(self,
                 problem: Problem,
                 perm: np.ndarray(shape=(1), dtype=int) = None
                 ):
        self.problem = problem

        if perm is None:
            self.perm = np.ndarray(shape=(self.problem.numberOfObjectives + self.problem.numberOfConstraints),
                                   dtype=int)
            for i in range(self.perm.size):
                self.perm[i] = i
        else:
            self.perm = perm

    def Calculate(self,
                  dataItem: SearchDataItem,
                  functionIndex: int,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                  ) -> SearchDataItem:
        """Compute selected function by number."""
        # ???
        dataItem.functionValues[self.perm[functionIndex]] = self.problem.Calculate(dataItem.point,
                                                                                   dataItem.functionValues[
                                                                                       self.perm[functionIndex]])
        return dataItem
