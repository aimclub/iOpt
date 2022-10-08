from abc import ABC, abstractmethod
from typing import List
import sys
import numpy as np
import system_architecture_design as sa

from enum import Enum

class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class BaseTaskForMethod:
    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
                 perm:np.ndarray(shape=(1), dtype=np.int)):
        self.problem = problem
        self.parameters = parameters
        self.perm = perm

    def calculate(self,
                  functionValue: sa.TrialPoint,
                  functionNumber:int,
                  type:TypeOfCalculation=TypeOfCalculation.FUNCTION
                  ) -> sa.TrialPoint:
        self.problem.calculate(functionValue.point,
                               functionValue.functionValues[self.perm[functionNumber]])
        """
        Compute selected function by number.
        :return: Calculated function value.
        """

class MethodPoint(sa.TrialPoint):
    __index: int = -2
    __discreteValueIndex: int = 0
    __z: np.double = sys.float_info.max
    def __init__(self,
                 y: sa.Point,
                 x: np.double = -1,
                 discreteValueIndex: int = 0
                ):
        self.point = y
        self.__x = x
        self.__discreteValueIndex = discreteValueIndex

    def getX(self)->np.double:
        pass

    def getY(self)->sa.Point:
        pass

    def setIndex(self, index:int):
        pass

    def getIndex(self)->int:
        pass

    def getDiscreteValueIndex(self)->int:
        pass

    def setZ(self, z:np.double):
        pass

    def getZ(self)->np.double:
        pass

class Evolvent:
    def __init__(self,
                 lowerBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double)=[],
                 upperBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double)=[],
                 numberOfFloatVariables:int = 1,
                 evolventDensity:int = 10
                 ):
        self.numberOfFloatVariables = numberOfFloatVariables
        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables
        self.evolventDensity = evolventDensity

    def GetImage( x:np.double) -> np.ndarray(shape=(1), dtype=np.double):
        """
        x->y
        """
        pass

    def GetInverseImage(y:np.ndarray(shape=(1), dtype=np.double) ) -> np.double:
        """
        y->x
        """
        pass

class Method:
    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
                 task: BaseTaskForMethod,
                 evolvent: Evolvent
                ):
        self.problem = problem
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent

    def FirstIteration(self):
        pass

    def CheckStopCondition(self):
        pass

    def CalculateIterationPoints(self) -> MethodPoint:
        pass

    def CalculateFunctionals(self, point: MethodPoint) -> MethodPoint:
        pass

    def RenewSearchData(self):
        pass

    def UpdateOptimum(self, point: MethodPoint):
        pass

    def FinalizeIteration(self):
        pass

class Process:
    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
                 task: BaseTaskForMethod,
                 evolvent: Evolvent,
                 method:Method
                 ):
        self.problem = problem
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.method = method

    def solve(self) -> sa.Solution:
        """
        Retrieve a solution with check of the stop conditions
        :return: Solution for the optimization problem
        """

    def performeGlobalIteration(self, number: int = 1):
        """
        :param number: The number of iterations of the global search
        """

    def performeLocalRefinement(self, number: int = 1):
        """
        :param number: The number of iterations of the local search
        """

    def getResults(self) -> sa.Solution:
        """
        :return: Return current solution for the optimization problem
        """