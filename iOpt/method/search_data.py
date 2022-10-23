from __future__ import annotations
from typing import List
import sys
import numpy as np
import queue

# from bintrees import AVLTree

from iOpt.trial import Point
from iOpt.trial import Trial
from iOpt.problem import Problem
from iOpt.solution import Solution


class SearchDataItem(Trial):
    __x: np.double = -1.0
    __index: int = -2
    __discreteValueIndex: int = 0
    __z: np.double = sys.float_info.max

    __leftPoint: SearchDataItem = None
    __rightPoint: SearchDataItem = None

    delta: np.double = -1.0

    localR: np.double = -1.0
    globalR: np.double = -1.0

    iterationNumber: int = -1

    def __init__(self,
                 y: Point,
                 x: np.double = -1,
                 discreteValueIndex: int = 0
                ):
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
    # пока без вытеснений
    __baseQueue: queue = queue.PriorityQueue()

    def __init__(self):
        self.__baseQueue = queue.PriorityQueue()

    def Clear(self):
        self.__baseQueue.queue.clear()

    def Insert(self, key: np.double, dataItem: SearchDataItem):
        # приоритет - значение характеристики
        # чтобы возвращало значение по убыванию
        # -1*dataItem.globalR
        self.__baseQueue.put((key, dataItem))

    def GetBestItem(self) -> SearchDataItem:
        # размер очереди = числу интервалов
        return self.__baseQueue.get()[1]

    def IsEmpty(self):
        return self.__baseQueue.empty()


class SearchData:
    # очереди характеристик
    __RLocalQueue: CharacteristicsQueue = CharacteristicsQueue()
    __RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue()
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    __allTrials: List = []
    __iterDataItem: SearchDataItem = None

    solution: Solution = None

    def __init__(self, problem: Problem):
        self.solution = Solution(problem)
        self.__allTrials = []
        self.__RLocalQueue = CharacteristicsQueue()
        self.__RGlobalQueue = CharacteristicsQueue()

    def ClearQueue(self):
        self.__RGlobalQueue.Clear()
        self.__RLocalQueue.Clear()

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если rigthDataItem == None то его необходимо найти по дереву __allTrials
    def InsertDataItem(self, newDataItem: SearchDataItem,
                       rigthDataItem: SearchDataItem = None):
        if rigthDataItem is None:
            rigthDataItem = self.FindDataItemByOneDimensionalPoint(newDataItem.GetX())

        newDataItem.SetLeft(rigthDataItem.GetLeft())
        newDataItem.SetRight(rigthDataItem)

        newDataItem.GetLeft().SetRight(newDataItem)
        newDataItem.GetRight().SetLeft(newDataItem)

        # вставка в лист
        self.__allTrials.append(newDataItem)

    def InsertFirstDataItem(self, leftDataItem : SearchDataItem,
                            rightDataItem: SearchDataItem):
        leftDataItem.SetRight(rightDataItem)
        rightDataItem.SetLeft(leftDataItem)
        self.__allTrials.append(leftDataItem)
        self.__allTrials.append(rightDataItem)
        self.__iterDataItem = leftDataItem

    # поиск покрывающего интервала
    # возвращает правую точку
    def FindDataItemByOneDimensionalPoint(self, x: np.double) -> SearchDataItem:
        # итерируемся по rightPoint от минимального элемента
        for item in self:
            # как только встретили интервал с большей точкой - останавливаемся
            if item.GetX() > x:
                rightDataItem = item
                break
        return rightDataItem

    def GetDataItemWithMaxGlobalR(self) -> SearchDataItem:
        if self.__RGlobalQueue.IsEmpty():
            self.RefillQueue()
        return self.__RGlobalQueue.GetBestItem()

    def GetDataItemWithMaxLocalR(self) -> SearchDataItem:
        if self.__RLocalQueue.IsEmpty():
            self.RefillQueue()
        return self.__RLocalQueue.GetBestItem()

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def RefillQueue(self):
        self.__RGlobalQueue.Clear()
        self.__RLocalQueue.Clear()

        for itr in self:
            self.__RGlobalQueue.Insert(-1 * itr.globalR, itr)
            self.__RLocalQueue.Insert(-1 * itr.localR, itr)

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
        self.curIter = self.__iterDataItem
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
