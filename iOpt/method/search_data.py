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
    __rigthPoint: SearchDataItem = None

    delta: np.double = -1.0

    localR: np.double = -1.0
    globalR: np.double = -1.0

    iterationNumber: int = -1

    # итератор по испытаниям
    def __next__(self):
        if self.__rigthPoint is not None:
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

    def SetRigth(self, point: SearchDataItem):
        self.__rightPoint = point

    def GetRigth(self) -> SearchDataItem:
        return self.__rightPoint

    def __lt__(self, other):
        return self.GetX() < other.GetX()


class CharacteristicsQueue:
    # пока без вытеснений
    __baseQueue: queue = queue.PriorityQueue()
    __typeR: int = 1

    def __init__(self, typeR: int):
        self.__typeR = typeR
        self.__baseQueue = queue.PriorityQueue()

    def Clear(self):
        self.__baseQueue.queue.clear()

    def Insert(self, dataItem: SearchDataItem):
        # typeR: 0 - localR, 1 - globalR
        # приоритет - значение характеристики
        # чтобы возвращало значение по убыванию
        # -1*dataItem.globalR
        if self.__typeR:
            data = (-1 * dataItem.globalR, dataItem)
        else:
            data = (-1 * dataItem.localR, dataItem)
        self.__baseQueue.put(data)

    def GetBestItem(self) -> SearchDataItem:
        # размер очереди = числу интервалов
        if self.IsEmpty():
            return None
        else:
            return self.__baseQueue.get()

    def IsEmpty(self):
        return self.__baseQueue.empty()


class SearchData:
    # очереди характеристик
    __RLocalQueue: CharacteristicsQueue = CharacteristicsQueue(typeR=0)
    __RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue(typeR=1)
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    __allTrials: List = []

    solution: Solution = None

    def __init__(self, problem: Problem):
        self.solution = Solution(problem)
        self.__allTrials = []
        self.__RLocalQueue = CharacteristicsQueue(typeR=0)
        self.__RGlobalQueue = CharacteristicsQueue(typeR=1)

    def ClearQueue(self):
        self.__RGlobalQueue.Clear()
        self.__RLocalQueue.Clear()

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если rigthDataItem == None то его необходимо найти по дереву __allTrials
    def InsertDataItem(self, newDataItem: SearchDataItem,
                       rigthDataItem: SearchDataItem = None):
        dimention = len(newDataItem.point.floatVariables)
        if self.GetCount() > 1:
            if rigthDataItem is None:
                rigthDataItem = self.FindDataItemByOneDimensionalPoint(newDataItem.GetX())

            newDataItem.SetLeft(rigthDataItem.GetLeft())
            newDataItem.SetRigth(rigthDataItem)

            newDataItem.GetLeft().SetRigth(newDataItem)
            newDataItem.GetRigth().SetLeft(newDataItem)

            # пересчет длин полученных интервалов
            newDataItem.delta = (newDataItem.GetX() -
                                 newDataItem.GetLeft().GetX()) ** (1 / dimention)
            newDataItem.GetRigth().delta = (newDataItem.GetRigth().GetX() -
                                            newDataItem.GetX()) ** (1 / dimention)
            # вставка в лист
            posInList = self.__allTrials.index(rigthDataItem)
            self.__allTrials.insert(posInList, newDataItem)
        else:
            if self.GetCount() == 1:
                newDataItem.delta = (newDataItem.GetX() - newDataItem.GetLeft().GetX()) ** (1 / dimention)
            self.__allTrials.append(newDataItem)

    # поиск покрывающего интервала
    # возвращает правую точку
    def FindDataItemByOneDimensionalPoint(self, x: np.double) -> SearchDataItem:
        posInList = 1
        count = len(self.__allTrials)
        if count > 2:
            left = 0
            right = count - 1
            while left <= right:
                mid = int((left + right) / 2)
                a = self.__allTrials[mid - 1].GetX()
                b = self.__allTrials[mid].GetX()
                if self.__allTrials[mid - 1].GetX() > x:
                    right = mid - 1
                elif self.__allTrials[mid].GetX() < x:
                    left = mid + 1
                else:
                    posInList = mid
                    break
        rightDataItem = self.__allTrials[posInList]
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

        for itr in self.__iter__():
            self.__RGlobalQueue.Insert(itr)
            self.__RLocalQueue.Insert(itr)

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
        return self.__allTrials.__iter__()

