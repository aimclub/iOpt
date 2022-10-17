from __future__ import annotations
from typing import List
import sys
import numpy as np
import queue

# from bintrees import AVLTree

from trial import Point
from trial import Trial
from problem import Problem
from solution import Solution


class SearchDataItem(Trial):
    __x: np.double = -1.0
    __index: int = -2
    __discreteValueIndex: int = 0
    __z: np.double = sys.float_info.max

    __leftPoint: SearchDataItem = None
    __rigthPoint: SearchDataItem = None

    delta: np.double = -1.0

    localR: np.double = -1.0
    globalR: np.double = -1.0

    iterationNumber: int = -1

    # итератор по испытаниям
    def __next__(self):
        if self.__rigthPoint != None:
            yield self.__rigthPoint

    def __init__(self,
                 y: Point,
                 x: np.double = -1,
                 discreteValueIndex: int = 0
                ):
        self.point = y
        self.__x = x
        self.__discreteValueIndex = discreteValueIndex

    def GetX(self) -> np.double:
        pass

    def GetY(self) -> Point:
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
    # очереди характеристик
    __RLocalQueue: CharacteristicsQueue = CharacteristicsQueue()
    __RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue()
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    __allTrials: List = []

    solution: Solution = None

    def __init__(self, problem: Problem):
        self.solution = Solution(problem)
        pass

    def ClearQueue(self):
        pass

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если rigthDataItem == None то его необходимо найти по дереву __allTrials
    def InsertDataItem(self, newDataItem: SearchDataItem, rigthDataItem: SearchDataItem = None):
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

