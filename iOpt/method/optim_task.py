from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt

from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem


class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class OptimizationTask:
    def __init__(self,
                 problem: Problem,
                 perm: Optional[npt.NDArray[np.int32]] = None
                 ) -> None:
        self.problem = problem

        if perm is None:
            self.perm = np.arange(self.problem.numberOfObjectives + self.problem.numberOfConstraints)
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
