from __future__ import annotations
import numpy as np
from enum import Enum

from search_data import SearchDataItem
from problem import Problem


class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class OptimizationTask:
    def __init__(self,
                 problem: Problem,
                 perm: np.ndarray(shape = (1), dtype = np.int)
                ):
        self.problem = problem
        self.perm = perm

    def Calculate(self,
                  dataItem: SearchDataItem,
                  functionIndex: int,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                 ) -> SearchDataItem:
        self.problem.Calculate(dataItem.point,
                               dataItem.functionValues[self.perm[functionIndex]])
        """
        Compute selected function by number.
        :return: Calculated function value.
        """

