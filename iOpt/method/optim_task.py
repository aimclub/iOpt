from __future__ import annotations
import numpy as np
from enum import Enum

from search_data import SearchDataItem
from iOpt.problem import Problem

#УДАЛИТЬ!
from iOpt.problems.rastrigin import Rastrigin
from iOpt.trial import Point


class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class OptimizationTask:
    def __init__(self,
                 problem: Problem,
                 perm: np.ndarray(shape=(1), dtype=np.int) = []
                 ):
        self.problem = problem

        if perm == []:
            self.perm = np.ndarray(shape=(self.problem.numberOfFloatVariables+self.problem.numberOfDisreteVariables), dtype=np.int)
            for i in range(self.perm.size):
                self.perm[i]=i
        else:
            self.perm = perm

    def Calculate(self,
                  dataItem: SearchDataItem,
                  functionIndex: int,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                  ) -> SearchDataItem:
        dataItem.functionValues[self.perm[functionIndex]] = self.problem.Calculate(dataItem.point,
                                                                                   dataItem.functionValues[
                                                                                       self.perm[functionIndex]])
        return dataItem
        """
        Compute selected function by number.
        :return: Calculated function value.
        """
if __name__ == "__main__":
    ras = Rastrigin(3)
    per = np.ndarray(shape=(1), dtype=np.int)
    #per =[1, 0, 2]
    #opT = OptimizationTask(ras, per)
    opT = OptimizationTask(ras)
    print(opT.perm[1])

    point = Point([0.0, 0.0, 0.0], [])
    sdi = SearchDataItem(point, -1, 0)
   # sdi.functionValues = np.ndarray(shape=(1), dtype=FunctionValue)
   # sdi[0] = FunctionValue()
    sdi = opT.Calculate(sdi, 0)

