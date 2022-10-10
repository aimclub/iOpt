from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import sys
import numpy as np
import system_architecture_design as sa
import queue
from bintrees import AVLTree

from enum import Enum


class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class BaseTaskForMethod:
    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
                 perm: np.ndarray(shape=(1), dtype=np.int)):
        self.problem = problem
        self.parameters = parameters
        self.perm = perm

    def calculate(self,
                  functionValue: sa.TrialPoint,
                  functionNumber: int,
                  type: TypeOfCalculation = TypeOfCalculation.FUNCTION
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

    __leftPoint: MethodPoint = None
    __rigthPoint: MethodPoint = None

    delta: np.double = -1

    localR: np.double = -1
    globalR: np.double = -1

    iterationNumber: int = -1

    def __init__(self,
                 y: sa.Point,
                 x: np.double = -1,
                 discreteValueIndex: int = 0
                 ):
        self.point = y
        self.__x = x
        self.__discreteValueIndex = discreteValueIndex

    def getX(self) -> np.double:
        pass

    def getY(self) -> sa.Point:
        pass

    def setIndex(self, index: int):
        pass

    def getIndex(self) -> int:
        pass

    def getDiscreteValueIndex(self) -> int:
        pass

    def setZ(self, z: np.double):
        pass

    def getZ(self) -> np.double:
        pass

    def getLeft(self) -> MethodPoint:
        return self.__leftPoint

    def setLeft(self, point: MethodPoint):
        self.__leftPoint = point

    def getRigth(self) -> MethodPoint:
        return self.__rigthPoint

    def setRigth(self, point: MethodPoint):
        self.__rigthPoint = point


class calcDelta:
    def __init__(self,
                 dim: int = 1,
                 ):
        self.dim

    def root(self, p1: MethodPoint, p2: MethodPoint) -> np.double:
        pass


class Evolvent:
    def __init__(self,
                 lowerBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double) = [],
                 upperBoundOfFloatVariables: np.ndarray(shape=(1), dtype=np.double) = [],
                 numberOfFloatVariables: int = 1,
                 evolventDensity: int = 10
                 ):
        self.numberOfFloatVariables = numberOfFloatVariables
        self.lowerBoundOfFloatVariables = lowerBoundOfFloatVariables
        self.upperBoundOfFloatVariables = upperBoundOfFloatVariables
        self.evolventDensity = evolventDensity

    def GetImage(x: np.double) -> np.ndarray(shape=(1), dtype=np.double):
        """
        x->y
        """
        pass

    def GetInverseImage(y: np.ndarray(shape=(1), dtype=np.double)) -> np.double:
        """
        y->x
        """
        pass


class CharacteristicsQueue:
    Q: queue = queue.PriorityQueue()
    def __init__(self):
        pass

    def ClearQueue(self):
        pass

    def InsertTrial(self, trial: MethodPoint):
        pass

    def getBestTrial(self) -> MethodPoint:
        pass


class SearchData:
    # очереди карактеристик
    __RLocalQueue: CharacteristicsQueue = CharacteristicsQueue()
    __RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue()
    # упорядоченное множество всех испытаний по X
    __allTrials: AVLTree = AVLTree()

    solution: sa.Solution = None

    def __init__(self, problem: sa.Problem):
        self.solution = sa.Solution(problem)
        pass

    def ClearQueue(self):
        pass

    # вставка точки без доп. инф.
    def InsertTrial(self, trial: MethodPoint):
        pass

    # вставка точки если знает правую точку
    def InsertTrial(self, newTrial: MethodPoint, rigthTrial: MethodPoint):
        pass

    def FindTrialByOneDimensionalPoint(self, x: np.double) -> MethodPoint:
        pass

    def GetTrialWithMaxGlobalR(self) -> MethodPoint:
        pass

    def GetTrialWithMaxLocalR(self) -> MethodPoint:
        pass

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def RefillQueue(self):
        pass

    # Возвращает текущее число интервалов в дереве
    def GetCount(self) -> int:
        pass


    def saveProgress(self, fileName: str):
        """
        :return:
        """

    def loadProgress(self, fileName: str):
        """
        :return:
        """


class Method:
    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
                 task: BaseTaskForMethod,
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

    def CalculateIterationPoints(self) -> MethodPoint:
        pass

    def CalculateFunctionals(self, point: MethodPoint) -> MethodPoint:
        pass

    def RenewSearchData(self, point: MethodPoint):
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
                 searchData: SearchData,
                 method: Method
                 ):
        self.problem = problem
        self.parameters = parameters
        self.task = task
        self.evolvent = evolvent
        self.searchData = searchData
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
