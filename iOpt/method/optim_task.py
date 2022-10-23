from __future__ import annotations
import numpy as np
from enum import Enum

from search_data import SearchDataItem
from iOpt.problem import Problem


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
		
		"""
		perm.shape = ???
		for i in ???
			perm[i]=i
		"""

    def Calculate(self,
                  dataItem: SearchDataItem,
                  functionIndex: int,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                 ) -> SearchDataItem:
		funcValue = self.problem.Calculate(dataItem.point,
                               dataItem.functionValues[self.perm[functionIndex]])		 
        return SearchDataItem(dataItem.point, funcValue)
        """
        Compute selected function by number.
        :return: Calculated function value.
        """

