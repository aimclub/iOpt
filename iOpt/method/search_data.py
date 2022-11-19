from __future__ import annotations
from typing import List
import sys
import numpy as np
import queue
import depq
from depq import DEPQ

# from bintrees import AVLTree

from iOpt.trial import Point, FunctionValue
from iOpt.trial import Trial
from iOpt.problem import Problem
from iOpt.solution import Solution


class SearchDataItem(Trial):
    __x: np.double
    __index: int = -2
    __discreteValueIndex: int = 0
    __z: np.double = sys.float_info.max

    __leftPoint: SearchDataItem = None
    __rightPoint: SearchDataItem = None

    delta: np.double = -1.0
    flag: int = 1

    # localR: np.double = -1.0
    globalR: np.double = -1.0

    iterationNumber: int = -1

    def __init__(self, y: Point, x: np.double,
                 functionValues: np.ndarray(shape=(1), dtype=FunctionValue) = [FunctionValue()],
                 discreteValueIndex: int = 0):
        super().__init__(point=y, functionValues=functionValues)
        self.point = y
        self.__x = x
        self.__discreteValueIndex = discreteValueIndex

    def GetX(self) -> np.double:
        return self.__x

    def GetY(self) -> Point:
        return self.point

    def GetDiscreteValueIndex(self) -> int:
        return self.__discreteValueIndex

    def SetIndex(self, index: int):
        self.__index = index

    def GetIndex(self) -> int:
        return self.__index

    def SetZ(self, z: np.double):
        self.__z = z

    def GetZ(self) -> np.double:
        return self.__z

    def SetLeft(self, point: SearchDataItem):
        self.__leftPoint = point

    def GetLeft(self) -> SearchDataItem:
        return self.__leftPoint

    def SetRight(self, point: SearchDataItem):
        self.__rightPoint = point

    def GetRight(self) -> SearchDataItem:
        return self.__rightPoint

    def __lt__(self, other):
        return self.GetX() < other.GetX()


class CharacteristicsQueue:
    __baseQueue: depq = DEPQ(iterable=None, maxlen=None)

    def __init__(self, maxlen: int):
        self.__baseQueue = DEPQ(maxlen=maxlen)

    def Clear(self):
        self.__baseQueue.clear()

    def Insert(self, key: np.double, dataItem: SearchDataItem):
        # приоритет - значение характеристики
        self.__baseQueue.insert(dataItem, key)

    def GetBestItem(self) -> SearchDataItem:
        return self.__baseQueue.popfirst()[0]

    def IsEmpty(self):
        return self.__baseQueue.is_empty()

    def GetMaxLen(self) -> int:
        return self.__baseQueue.maxlen

    def GetLen(self) -> int:
        return len(self.__baseQueue)


class SearchData:
    # очереди характеристик
    __RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue(None)
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    __allTrials: List = []
    __firstDataItem: SearchDataItem = None

    solution: Solution = None

    def __init__(self, problem: Problem, maxlen: int = None):
        self.solution = Solution(problem)
        self.__allTrials = []
        self.__RGlobalQueue = CharacteristicsQueue(maxlen)

    def ClearQueue(self):
        self.__RGlobalQueue.Clear()

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если rigthDataItem == None то его необходимо найти по дереву __allTrials
    def InsertDataItem(self, newDataItem: SearchDataItem,
                       rigthDataItem: SearchDataItem = None):
        if rigthDataItem is None:
            rigthDataItem = self.FindDataItemByOneDimensionalPoint(newDataItem.GetX())

        newDataItem.SetLeft(rigthDataItem.GetLeft())
        rigthDataItem.SetLeft(newDataItem)
        newDataItem.SetRight(rigthDataItem)
        newDataItem.GetLeft().SetRight(newDataItem)

        self.__allTrials.append(newDataItem)

        self.__RGlobalQueue.Insert(newDataItem.globalR, newDataItem)
        # всегда ли делается?
        if rigthDataItem.flag == 0:
            self.__RGlobalQueue.Insert(rigthDataItem.globalR, rigthDataItem)

    def InsertFirstDataItem(self, leftDataItem: SearchDataItem,
                            rightDataItem: SearchDataItem):
        leftDataItem.SetRight(rightDataItem)
        rightDataItem.SetLeft(leftDataItem)

        self.__allTrials.append(leftDataItem)
        self.__allTrials.append(rightDataItem)

        self.__RGlobalQueue.Insert(leftDataItem.globalR, leftDataItem)
        self.__RGlobalQueue.Insert(rightDataItem.globalR, rightDataItem)

        self.__firstDataItem = leftDataItem

    # поиск покрывающего интервала
    # возвращает правую точку
    def FindDataItemByOneDimensionalPoint(self, x: np.double) -> SearchDataItem:
        # итерируемся по rightPoint от минимального элемента
        for item in self:
            if item.GetX() > x:
                return item
        return None

    def GetDataItemWithMaxGlobalR(self) -> SearchDataItem:
        if self.__RGlobalQueue.IsEmpty():
            self.RefillQueue()
        bestItem = self.__RGlobalQueue.GetBestItem()
        bestItem.flag = 0
        return bestItem

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def RefillQueue(self):
        self.__RGlobalQueue.Clear()

        for itr in self:
            self.__RGlobalQueue.Insert(itr.globalR, itr)

    # Возвращает текущее число интервалов в дереве
    def GetCount(self) -> int:
        return len(self.__allTrials)

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
        # return self.__allTrials.min_item()[1]
        self.curIter = self.__firstDataItem
        if self.curIter is None:
            raise StopIteration
        else:
            return self

    def __next__(self):
        if self.curIter is None:
            raise StopIteration
        else:
            tmp = self.curIter
            self.curIter = self.curIter.GetRight()
            return tmp
