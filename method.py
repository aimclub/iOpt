from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import sys
import numpy as np
import system_architecture_design as sa
import queue
from bintrees import AVLTree

from enum import Enum


class SearchDataItem(sa.Trial):
    __index: int = -2
    __discreteValueIndex: int = 0
    __z: np.double = sys.float_info.max

    __leftPoint: SearchDataItem = None
    __rigthPoint: SearchDataItem = None

    delta: np.double = -1

    localR: np.double = -1
    globalR: np.double = -1

    iterationNumber: int = -1

    # итератор по испытаниям
    def __next__(self):
        if self.__rigthPoint != None:
            yield self.__rigthPoint

    def __init__(self,
                 y: sa.Point,
                 x: np.double = -1,
                 discreteValueIndex: int = 0
                ):
        self.point = y
        self.__x = x
        self.__discreteValueIndex = discreteValueIndex

    def GetX(self) -> np.double:
        pass

    def GetY(self) -> sa.Point:
        pass

    def GetDiscreteValueIndex(self) -> int:
        pass

    def SetIndex(self, index: int):
        pass

    def GetIndex(self) -> int:
        pass

    def SetZ(self, z: np.double):
        pass

    def GetZ(self) -> np.double:
        pass

    def SetLeft(self, point: SearchDataItem):
        self.__leftPoint = point

    def GetLeft(self) -> SearchDataItem:
        return self.__leftPoint

    def SetRigth(self, point: SearchDataItem):
        self.__rigthPoint = point

    def GetRigth(self) -> SearchDataItem:
        return self.__rigthPoint


class TypeOfCalculation(Enum):
    FUNCTION = 1
    CONVOLUTION = 2


class OptimizationTask:
    def __init__(self,
                 problem: sa.Problem,
                 #arameters: sa.SolverParameters,
                 perm: np.ndarray(shape=(1), dtype=np.int)
                ):
        self.problem = problem
        #self.parameters = parameters
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

    def GetImage(self,
                 x: np.double
                ) -> np.ndarray(shape=(1), dtype=np.double):
        """
        x->y
        """
        pass

    def GetInverseImage(self,
                        y: np.ndarray(shape=(1), dtype=np.double)
                       ) -> np.double:
        """
        y->x
        """
        pass


class CharacteristicsQueue:
    __baseQueue: queue = queue.PriorityQueue()

    def __init__(self):
        pass

    def Clear(self):
        pass

    def Insert(self, dataItem: SearchDataItem):
        pass

    def GetBestItem(self) -> SearchDataItem:
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

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если rigthDataItem == None то его необходимо найти по дереву __allTrials
    def InsertDataItem(self, newDataItem: SearchDataItem, rigthDataItem: SearchDataItem=None):
        pass

    def FindDataItemByOneDimensionalPoint(self, x: np.double) -> SearchDataItem:
        pass

    def GetDataItemWithMaxGlobalR(self) -> SearchDataItem:
        pass

    def GetDataItemWithMaxLocalR(self) -> SearchDataItem:
        pass

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def RefillQueue(self):
        pass

    # Возвращает текущее число интервалов в дереве
    def GetCount(self) -> int:
        pass

    def SaveProgress(self, fileName: str):
        """
        :return:
        """

    def LoadProgress(self, fileName: str):
        """
        :return:
        """

    def __iter__(self):
        # вернуть самую левую точку из дерева (ниже код проверить!)
        return self.__allTrials.min_item()[1]


class Listener:
    def BeforeMethodStart(self, searchData: SearchData):
        pass

    def OnEndIteration(self, searchData: SearchData):
        pass

    def OnMethodStop(self, searchData: SearchData):
        pass

    def OnRefrash(self, searchData: SearchData):
        pass


class FunctionPainter:
    def __init__(self, searchData: SearchData):
        self.searchData = searchData

    def Paint(self):
        pass


# пример слушателя
class PaintListener(Listener):
    # нарисовать все точки испытаний
    def OnMethodStop(self, searchData: SearchData):
        fp = FunctionPainter(searchData)
        fp.Paint()
        pass


class Method:
    stop: bool = False

    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
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

    def CalculateIterationPoints(self) -> SearchDataItem:
        pass

    def CalculateFunctionals(self, point: SearchDataItem) -> SearchDataItem:
        pass

    def RenewSearchData(self, point: SearchDataItem):
        pass

    def UpdateOptimum(self, point: SearchDataItem):
        pass

    def FinalizeIteration(self):
        pass


class Process:
    __listeners: List[Listener] = []

    def __init__(self,
                 problem: sa.Problem,
                 parameters: sa.SolverParameters,
                 task: OptimizationTask,
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

    def Solve(self) -> sa.Solution:
        """
        Retrieve a solution with check of the stop conditions
        :return: Solution for the optimization problem
        """

    def DoGlobalIteration(self, number: int = 1):
        """
        :param number: The number of iterations of the global search
        """

    def DoLocalRefinement(self, number: int = 1):
        """
        :param number: The number of iterations of the local search
        """

    def GetResults(self) -> sa.Solution:
        """
        :return: Return current solution for the optimization problem
        """

    def RefreshListener(self):
        pass

    def AddListener(self, listener: Listener):
        self.__listeners.append(listener)
