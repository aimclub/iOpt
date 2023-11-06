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
            self.perm = np.ndarray(shape=(self.problem.number_of_objectives +
                                          self.problem.number_of_constraints),
                                   dtype=int)
            for i in range(self.perm.size):
                self.perm[i] = i
        else:
            self.perm = perm

    def calculate(self,
                  data_item: SearchDataItem,
                  function_index: int,
                  calculation_type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                  ) -> SearchDataItem:
        """Compute selected function by number"""
        # ???
        data_item.function_values[self.perm[function_index]] = self.problem.calculate(data_item.point,
                                                                                      data_item.function_values[
                                                                                          self.perm[function_index]])
        if not np.isfinite(data_item.function_values[self.perm[function_index]].value):
            raise Exception("Infinity values")

        return data_item
