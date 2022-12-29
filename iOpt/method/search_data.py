from __future__ import annotations

import sys

import numpy as np
from depq import DEPQ

from iOpt.problem import Problem
from iOpt.solution import Solution
from iOpt.trial import Point, FunctionValue
from iOpt.trial import Trial


# from bintrees import AVLTree


class SearchDataItem(Trial):
    """
        Класс SearchDataItem предназначен для хранения поисковой информации, представляющей собой
        интервал с включенной правой точкой, а так же ссылками на соседние интервалы. SearchDataItem
        является наследником от класса Trial.
    """

    def __init__(self, y: Point, x: np.double,
                 functionValues: np.ndarray(shape=(1), dtype=FunctionValue) = [FunctionValue()],
                 discreteValueIndex: int = 0):
        """
        Конструктор класса SearchDataItem

        :param y: Точка испытания в исходной N-мерной области поиска
        :param x: Отображении точки испытания y на отрезок [0, 1]
        :param functionValues: Вектор значений функций (целевой функции и функций ограничений)
        :param discreteValueIndex: Дискретный параметр
        """
        super().__init__(point=y, functionValues=functionValues)
        self.point = y
        self.__x = x
        self.__discreteValueIndex = discreteValueIndex
        self.__index: int = -2
        self.__z: np.double = sys.float_info.max
        self.__leftPoint: SearchDataItem = None
        self.__rightPoint: SearchDataItem = None
        self.delta: np.double = -1.0
        self.globalR: np.double = -1.0
        self.localR: np.double = -1.0
        self.iterationNumber: int = -1

    def GetX(self) -> np.double:
        """
        Метод позволяет получить правую точку поискового интервала, где :math:`x\in[0, 1]`.

        :return: Значение правой точки интервала
        """
        return self.__x

    def GetY(self) -> Point:
        """
        Метод позволяет получить N-мерную точку испытания исходной области поиска.

        :return: Значение N-мерной точки испытания
        """
        return self.point

    def GetDiscreteValueIndex(self) -> int:
        """
        Метод позволяет получить дискретный параметр.

        :return: Значение дискретного параметра
        """
        return self.__discreteValueIndex

    def SetIndex(self, index: int):
        """
        Метод позволяет задать значение индекса последнего выполненного ограничения
        для индексной схемы.

        :param index: Индекс ограничения
        """
        self.__index = index

    def GetIndex(self) -> int:
        """
        Метод позволяет получить значение индекса последнего выполненного ограничения
        для индексной схемы.

        :return: Значение индекса
        """
        return self.__index

    def SetZ(self, z: np.double):
        """
        Метод позволяет задать значение функции для заданного индекса.

        :param z: Значение функции
        """
        self.__z = z

    def GetZ(self) -> np.double:
        """
        Метод позволяет получить значение функции для заданного индекса.

        :return: Значение функции для index
        """
        return self.__z

    def SetLeft(self, point: SearchDataItem):
        """
        Метод позволяет задать левый интервал для исходного.

        :param point: Левый интервал
        """
        self.__leftPoint = point

    def GetLeft(self) -> SearchDataItem:
        """
        Метод позволяет получить левый интервал для исходного.

        :return: Значение левого интервала
        """
        return self.__leftPoint

    def SetRight(self, point: SearchDataItem):
        """
        Метод позволяет задать правый интервал для исходного.

        :param point: Правый интервал
        """
        self.__rightPoint = point

    def GetRight(self) -> SearchDataItem:
        """
       Метод позволяет получить правый интервал для исходного.

       :return: Значение правого интервала
       """
        return self.__rightPoint

    def __lt__(self, other):
        """
       Метод переопределяет оператор сравнения < для двух интервалов.
       :param other: Второй интервал
       :return: Значение true - если правая точка исходного интервала меньше
       правой точки второго, иначе - false.
       """
        return self.GetX() < other.GetX()


class CharacteristicsQueue:
    """
    Класс CharacteristicsQueue предназначен для хранения приоритетной очереди
    характеристик с вытеснением.
    """

    def __init__(self, maxlen: int):
        """
        Конструктор класса CharacteristicsQueue

        :param maxlen: Максимальный размер очереди
        """
        self.__baseQueue = DEPQ(iterable=None, maxlen=maxlen)

    def Clear(self):
        """
        Метод позволяет очистить очередь
        """
        self.__baseQueue.clear()

    def Insert(self, key: np.double, dataItem: SearchDataItem):
        """
        Метод добавляет поисковый интервал с указанным приоритетом.
        Приоритетом является значение характеристики на данном интервале.

        :param key: Приоритет поискового интервала
        :param dataItem: Вставляемый интервал
        """
        self.__baseQueue.insert(dataItem, key)

    def GetBestItem(self) -> (SearchDataItem, np.double):
        """
        Метод позволяет получить интервал с лучшей характеристикой

        :return: Кортеж: интервал с лучшей характеристикой, приоритет интервала в очереди
        """
        return self.__baseQueue.popfirst()

    def IsEmpty(self):
        """
        Метод позволяет сделать проверку на пустоту очереди.

        :return: Значение true если очередь пуста, иначе false
        """
        return self.__baseQueue.is_empty()

    def GetMaxLen(self) -> int:
        """
        Метод позволяет получить максимальный размер очереди.

        :return: Значение максимального размера очереди
        """
        return self.__baseQueue.maxlen

    def GetLen(self) -> int:
        """
        Метод позволяет получить текущий размер очереди.

        :return: Значение текущего размера очереди
        """
        return len(self.__baseQueue)


class SearchData:
    """
    Класс SearchData предназначен для хранения множества всех интервалов, исходной задачи
    и приоритетной очереди глобальных характеристик.
    """

    # очереди характеристик
    # _RGlobalQueue: CharacteristicsQueue = CharacteristicsQueue(None)
    # упорядоченное множество всех испытаний по X
    # __allTrials: AVLTree = AVLTree()
    # _allTrials: List = []
    # __firstDataItem:

    # solution: Solution = None

    def __init__(self, problem: Problem, maxlen: int = None):
        """
        Конструктор класса SearchData

        :param problem: Информация об исходной задаче
        :param maxlen: Максимальный размер очереди
        """
        self.solution = Solution(problem)
        self._allTrials = []
        self._RGlobalQueue = CharacteristicsQueue(maxlen)
        self.__firstDataItem: SearchDataItem = None

    def ClearQueue(self):
        """
        Метод позволяет очистить очередь характеристик
        """
        self._RGlobalQueue.Clear()

    # вставка точки если знает правую точку
    # в качестве интервала используем [i-1, i]
    # если rightDataItem == None то его необходимо найти по дереву _allTrials
    def InsertDataItem(self, newDataItem: SearchDataItem,
                       rightDataItem: SearchDataItem = None):
        """
        Метод позволяет добавить новый интервал испытаний в список всех проведенных испытаний
        и приоритетную очередь характеристик.

        :param newDataItem: Новый интервал испытаний
        :param rightDataItem: Покрывающий интервал, является правым интервалом для newDataItem
        """
        flag = True
        if rightDataItem is None:
            rightDataItem = self.FindDataItemByOneDimensionalPoint(newDataItem.GetX())
            flag = False

        newDataItem.SetLeft(rightDataItem.GetLeft())
        rightDataItem.SetLeft(newDataItem)
        newDataItem.SetRight(rightDataItem)
        newDataItem.GetLeft().SetRight(newDataItem)

        self._allTrials.append(newDataItem)

        self._RGlobalQueue.Insert(newDataItem.globalR, newDataItem)
        if flag:
            self._RGlobalQueue.Insert(rightDataItem.globalR, rightDataItem)

    def InsertFirstDataItem(self, leftDataItem: SearchDataItem,
                            rightDataItem: SearchDataItem):
        """
        Метод позволяет добавить пару интервалов испытаний на первой итерации AGP.

        :param leftDataItem: Левый интервал для rightDataItem
        :param rightDataItem: Правый интервал для leftDataItem
        """
        leftDataItem.SetRight(rightDataItem)
        rightDataItem.SetLeft(leftDataItem)

        self._allTrials.append(leftDataItem)
        self._allTrials.append(rightDataItem)

        self.__firstDataItem = leftDataItem

    # поиск покрывающего интервала
    # возвращает правую точку
    def FindDataItemByOneDimensionalPoint(self, x: np.double) -> SearchDataItem:
        """
        Метод позволяет найти покрывающий интервал для полученной точки x.

        :param x: Правая точка интервала
        :return: Правая точка покрывающего интервала
        """
        # итерируемся по rightPoint от минимального элемента
        for item in self:
            if item.GetX() > x:
                return item
        return None

    def GetDataItemWithMaxGlobalR(self) -> SearchDataItem:
        """
        Метод позволяет получить интервал с лучшим значением глобальной характеристики.

        :return: Значение интервала с лучшей глобальной характеристикой
        """
        if self._RGlobalQueue.IsEmpty():
            self.RefillQueue()
        return self._RGlobalQueue.GetBestItem()[0]

    # Перезаполнение очереди (при ее опустошении или при смене оценки константы Липшица)
    def RefillQueue(self):
        """
        Метод позволяет перезаполнить очередь глобальных характеристик, например, при ее опустошении
        или при смене оценки константы Липшица.

        """
        self._RGlobalQueue.Clear()
        for itr in self:
            self._RGlobalQueue.Insert(itr.globalR, itr)

    # Возвращает текущее число интервалов в дереве
    def GetCount(self) -> int:
        """
        Метод позволяет получить текущее число интервалов в списке.

        :return: Значение числа интервалов в списке
        """
        return len(self._allTrials)

    def GetLastItem(self) -> SearchDataItem:
        """
        Метод позволяет получить последний добавленный интервал в список.

        :return: Значение последнего добавленного интервала
        """
        try:
            return self._allTrials[-1]
        except Exception:
            print("GetLastItem: List is empty")

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
        # return self._allTrials.min_item()[1]
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


class SearchDataDualQueue(SearchData):
    """
    Класс SearchDataDualQueue является наследником класса SearchData. Предназначен
      для хранения множества всех интервалов, исходной задачи и двух приоритетных очередей
      для глобальных и локальных характеристик.

    """

    def __init__(self, problem: Problem, maxlen: int = None):
        """
        Конструктор класса SearchDataDualQueue

        :param problem: Информация об исходной задаче
        :param maxlen: Максимальный размер очереди
        """
        super().__init__(problem, maxlen)
        self.__RLocalQueue = CharacteristicsQueue(maxlen)

    def ClearQueue(self):
        """
        Метод позволяет очистить очереди характеристик
        """
        self._RGlobalQueue.Clear()
        self.__RLocalQueue.Clear()

    def InsertDataItem(self, newDataItem: SearchDataItem,
                       rightDataItem: SearchDataItem = None):
        """
        Метод позволяет добавить новый интервал испытаний в список всех проведенных испытаний
          и приоритетные очереди глобальных и локальных характеристик.

        :param newDataItem: Новый интервал испытаний
        :param rightDataItem: Покрывающий интервал, является правым интервалом для newDataItem
        """
        flag = True
        if rightDataItem is None:
            rightDataItem = self.FindDataItemByOneDimensionalPoint(newDataItem.GetX())
            flag = False

        newDataItem.SetLeft(rightDataItem.GetLeft())
        rightDataItem.SetLeft(newDataItem)
        newDataItem.SetRight(rightDataItem)
        newDataItem.GetLeft().SetRight(newDataItem)

        self._allTrials.append(newDataItem)

        self._RGlobalQueue.Insert(newDataItem.globalR, newDataItem)
        self.__RLocalQueue.Insert(newDataItem.localR, newDataItem)
        if flag:
            self._RGlobalQueue.Insert(rightDataItem.globalR, rightDataItem)
            self.__RLocalQueue.Insert(rightDataItem.localR, rightDataItem)

    def GetDataItemWithMaxGlobalR(self) -> SearchDataItem:
        """
       Метод позволяет получить интервал с лучшим значением глобальной характеристики.

       :return: Значение интервала с лучшей глобальной характеристикой
       """
        if self._RGlobalQueue.IsEmpty():
            self.RefillQueue()
        bestItem = self._RGlobalQueue.GetBestItem()
        while bestItem[1] != bestItem[0].globalR:
            if self._RGlobalQueue.IsEmpty():
                self.RefillQueue()
            bestItem = self._RGlobalQueue.GetBestItem()
        return bestItem[0]

    def GetDataItemWithMaxLocalR(self) -> SearchDataItem:
        """
       Метод позволяет получить интервал с лучшим значением локальной характеристики.

       :return: Значение интервала с лучшей локальной характеристикой
       """
        if self.__RLocalQueue.IsEmpty():
            self.RefillQueue()
        bestItem = self.__RLocalQueue.GetBestItem()
        while bestItem[1] != bestItem[0].localR:
            if self.__RLocalQueue.IsEmpty():
                self.RefillQueue()
            bestItem = self.__RLocalQueue.GetBestItem()
        return bestItem[0]

    def RefillQueue(self):
        """
       Метод позволяет перезаполнить очереди глобальных и локальных характеристик, например,
         при их опустошении или при смене оценки константы Липшица.

       """
        self.ClearQueue()
        for itr in self:
            self._RGlobalQueue.Insert(itr.globalR, itr)
            self.__RLocalQueue.Insert(itr.localR, itr)
