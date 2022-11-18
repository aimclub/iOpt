import unittest

import iOpt.method.search_data
import iOpt.trial

from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point

from iOpt.method.search_data import CharacteristicsQueue
from iOpt.method.search_data import SearchData


class TestSearchDataItem(unittest.TestCase):
    # setUp method is overridden from the parent class SearchDataItem
    def setUp(self):
        self.point = Point([0.1, 0.5], ["a", "b"])
        self.searchDataItem = SearchDataItem(self.point, 0.3, 0)

    def test_Init(self):
        self.assertEqual(self.searchDataItem.point.floatVariables, [0.1, 0.5])

    def test_GetX(self):
        self.assertEqual(self.searchDataItem.GetX(), 0.3)

    def test_GetY(self):
        self.assertEqual((self.searchDataItem.GetY().floatVariables,
                          self.searchDataItem.GetY().discreteVariables),
                         ([0.1, 0.5], ['a', 'b']))

    def test_GetDiscreteValueIndex(self):
        self.assertEqual(self.searchDataItem.GetDiscreteValueIndex(), 0)

    def test_SetIndex(self):
        self.searchDataItem.SetIndex(1)
        self.assertEqual(self.searchDataItem.GetIndex(), 1)

    def test_SetZ(self):
        self.searchDataItem.SetZ(-4.0)
        self.assertEqual(self.searchDataItem.GetZ(), -4.0)

    def test_SetLeft(self):
        pointl = Point([0.3, 0.78], ["c", "d"])
        leftDataItem = SearchDataItem(pointl, 0.6,  None, 1)
        self.searchDataItem.SetLeft(leftDataItem)

        self.assertEqual(self.searchDataItem.GetLeft().GetX(), 0.6)
        self.assertEqual(self.searchDataItem.GetLeft().GetDiscreteValueIndex(), 1)
        self.assertEqual(self.searchDataItem.GetLeft().GetY().floatVariables, [0.3, 0.78])
        self.assertEqual(self.searchDataItem.GetLeft().GetY().discreteVariables, ['c', 'd'])

    def test_SetRigth(self):
        pointr = Point([-0.3, 0.78], ["e", "f"])
        rightDataItem = SearchDataItem(pointr, 0.16,  None, 0)
        self.searchDataItem.SetRight(rightDataItem)

        self.assertEqual(self.searchDataItem.GetRight().GetX(), 0.16)
        self.assertEqual(self.searchDataItem.GetRight().GetDiscreteValueIndex(), 0)
        self.assertEqual(self.searchDataItem.GetRight().GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(self.searchDataItem.GetRight().GetY().discreteVariables, ['e', 'f'])


class TestCharacteristicsQueue(unittest.TestCase):
    def setUp(self):
        self.characteristicsQueueGlobalR = CharacteristicsQueue(maxlen=3)
        self.characteristicsQueueLocalR = CharacteristicsQueue(maxlen=3)

    def test_Insert(self):
        point = Point([1.03, 0.5], ["k", "m"])
        dataItem = SearchDataItem(point, 0,  None, 1)
        dataItem.globalR = 2.56
        dataItem.localR = -0.2
        try:
            self.characteristicsQueueLocalR.Insert(dataItem.localR, dataItem)
            self.characteristicsQueueGlobalR.Insert(dataItem.globalR, dataItem)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Insert'," \
                          f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_InsertWithEqualR(self):

        point = Point([-0.3, 0.78], ["e", "m"])
        dataItem = SearchDataItem(point, 0.2,  None, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2

        point = Point([-0.5, 0.07], ["a", "f"])
        dataItem = SearchDataItem(point, 0.6, None, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2
        try:
            self.characteristicsQueueLocalR.Insert(dataItem.localR, dataItem)
            self.characteristicsQueueGlobalR.Insert(dataItem.globalR, dataItem)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Insert'," \
                          f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_Clear(self):
        try:
            self.characteristicsQueueLocalR.Clear()
            self.characteristicsQueueGlobalR.Clear()
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Clear'," \
                          f"'self.characteristicsQueueGlobalR.Clear' raised an exception{exc}"
        self.assertEqual(self.characteristicsQueueLocalR.IsEmpty(), True)
        self.assertEqual(self.characteristicsQueueGlobalR.IsEmpty(), True)

    def test_CanGetBestItem(self):
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0.2, None, 1)
        dataItem1.globalR = 4.76
        dataItem1.localR = 3.2
        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)

        point2 = Point([-0.6, 0.7], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, None, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = -3.8
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)

        getDataItem = self.characteristicsQueueGlobalR.GetBestItem()

        self.assertEqual(getDataItem.GetX(), 0.05)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.6, 0.7])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['e', 'f'])

    def test_CanGetBestItmeWithEqualR(self):
        point1 = Point([-0.6, 0.7], ["a", "f"])
        dataItem1 = SearchDataItem(point1, 0.05, None, 2)
        dataItem1.globalR = 5.0
        dataItem1.localR = 0.56
        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)
        self.characteristicsQueueLocalR.Insert(dataItem1.localR, dataItem1)

        point2 = Point([-0.3, 0.78], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.4, None, 1)
        dataItem2.globalR = 5.0
        dataItem2.localR = 1.2
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)
        self.characteristicsQueueLocalR.Insert(dataItem2.localR, dataItem2)

        getDataItemG = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItemL = self.characteristicsQueueLocalR.GetBestItem()

        self.assertEqual(getDataItemG.GetX(), 0.05)
        self.assertEqual(getDataItemG.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItemG.globalR, 5.0)
        self.assertEqual(getDataItemG.GetY().floatVariables, [-0.6, 0.7])
        self.assertEqual(getDataItemG.GetY().discreteVariables, ['a', 'f'])

        self.assertEqual(getDataItemL.GetX(), 0.4)
        self.assertEqual(getDataItemL.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItemL.localR, 1.2)
        self.assertEqual(getDataItemL.GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(getDataItemL.GetY().discreteVariables, ['e', 'f'])

    def test_GetLen(self):
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0.2, None, 1)
        dataItem1.globalR = 4.76
        dataItem1.localR = 3.2
        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)

        point2 = Point([-0.6, 0.7], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, None, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = -3.8
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(), 2)
        self.assertEqual(self.characteristicsQueueLocalR.GetLen(), 0)

    def test_GetMaxLen(self):
        self.assertEqual(self.characteristicsQueueGlobalR.GetMaxLen(), 3)
        self.assertEqual(self.characteristicsQueueLocalR.GetMaxLen(), 3)

    def test_NotAddItemWithLowerPriority(self):
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0.2, None, 1)
        dataItem1.globalR = 4.76
        dataItem1.localR = 3.2
        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)

        point2 = Point([-0.6, 0.7], ["s", "k"])
        dataItem2 = SearchDataItem(point2, 0.05, None, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = -3.8
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)

        point3 = Point([1.08, 4.56], ["a", "b"])
        dataItem3 = SearchDataItem(point3, 0.7, None, 1)
        dataItem3.globalR = 3.05
        dataItem3.localR = -0.12
        self.characteristicsQueueGlobalR.Insert(dataItem3.globalR, dataItem3)

        point4 = Point([0.076, 2.7], ["c", "d"])
        dataItem4 = SearchDataItem(point4, 0.5, None, 1)
        dataItem4.globalR = 1.5
        dataItem4.localR = 3.8
        self.characteristicsQueueGlobalR.Insert(dataItem4.globalR, dataItem4)

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(),
                         self.characteristicsQueueGlobalR.GetMaxLen())

        getDataItem1 = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItem2 = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItem3 = self.characteristicsQueueGlobalR.GetBestItem()

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(), 0)

        self.assertEqual(getDataItem1.GetX(), 0.05)
        self.assertEqual(getDataItem1.globalR, 5.0)

        self.assertEqual(getDataItem2.GetX(), 0.2)
        self.assertEqual(getDataItem2.globalR, 4.76)

        self.assertEqual(getDataItem3.GetX(), 0.7)
        self.assertEqual(getDataItem3.globalR, 3.05)

    def test_AddItemWithHightPriority(self):
        # maxlen = 3
        point1 = Point([-0.6, 0.7], ["a", "f"])
        dataItem1 = SearchDataItem(point1, 0.05, None, 2)
        dataItem1.globalR = 4.10
        dataItem1.localR = 0.56
        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)

        point2 = Point([-0.3, 0.78], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.74, None, 1)
        dataItem2.globalR = 2.50
        dataItem2.localR = 1.2
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)

        point3 = Point([1.5, 3.2], ["r", "s"])
        dataItem3 = SearchDataItem(point3, 0.12, None, 0)
        dataItem3.globalR = 3.23
        dataItem3.localR = 0.56
        self.characteristicsQueueGlobalR.Insert(dataItem3.globalR, dataItem3)

        point4 = Point([2.67, -0.78], ["h", "k"])
        dataItem4 = SearchDataItem(point4, 0.4, None, 1)
        dataItem4.globalR = 3.0
        dataItem4.localR = -0.89
        self.characteristicsQueueGlobalR.Insert(dataItem4.globalR, dataItem4)

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(),
                         self.characteristicsQueueGlobalR.GetMaxLen())
        getDataItem1 = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItem2 = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItem3 = self.characteristicsQueueGlobalR.GetBestItem()

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(), 0)
        self.assertEqual(getDataItem1.GetX(), 0.05)
        self.assertEqual(getDataItem1.globalR, 4.10)

        self.assertEqual(getDataItem2.GetX(), 0.12)
        self.assertEqual(getDataItem2.globalR, 3.23)

        self.assertEqual(getDataItem3.GetX(), 0.4)
        self.assertEqual(getDataItem3.globalR, 3.0)


class TestSearchData(unittest.TestCase):
    def setUp(self):
        self.searchData = SearchData(None)

    def test_ClearQueue(self):
        try:
            self.searchData.ClearQueue()
        except Exception as exc:
            assert False, f"'self.searchData.ClearQueue' raised an exception{exc}"

    def test_InsertDataItemFirst(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        leftdataItem = SearchDataItem(point, 0, None, 1)
        leftdataItem.globalR = 4.76
        leftdataItem.localR = 3.2

        point = Point([-0.3, 0.78], ["a", "b"])
        rightdataItem = SearchDataItem(point, 1, None, 1)
        rightdataItem.globalR = -3.2
        rightdataItem.localR = 1.03

        try:
            self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)
        except Exception as exc:
            assert False, f"' self.searchData.InsertDataItemFirst' raised an exception{exc}"

    def test_InsertDataItemWithRight(self):
        point = Point([1.4, 3.078], ["c", "f"])
        dataItem = SearchDataItem(point, 0.5, None, 1)
        dataItem.globalR = -0.9
        dataItem.localR = 0.5

        # rightPoint
        point = Point([-0.3, 0.78], ["a", "b"])
        rightdataItem = SearchDataItem(point, 1, None, 1)
        rightdataItem.globalR = -3.2
        rightdataItem.localR = 1.03
        # leftPoint
        point = Point([-0.3, 0.78], ["e", "f"])
        leftdataItem = SearchDataItem(point, 0, None, 1)
        leftdataItem.globalR = 4.76
        leftdataItem.localR = 3.2

        self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)

        try:
            self.searchData.InsertDataItem(dataItem, rightdataItem)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_CanUseIteration(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        leftdataItem = SearchDataItem(point, 0, None, 1)
        leftdataItem.globalR = 4.76
        leftdataItem.localR = 3.2

        point = Point([-0.3, 0.78], ["a", "b"])
        rightdataItem = SearchDataItem(point, 1, None, 1)
        rightdataItem.globalR = -3.2
        rightdataItem.localR = 1.03

        self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)

        getData = []
        try:
            for item in self.searchData:
                getData.append(item.GetX())
        except Exception as exc:
            assert False, f"'item.GetX' raised an exception{exc}"

        self.assertEqual(getData[0], leftdataItem.GetX())
        self.assertEqual(getData[1], rightdataItem.GetX())

    def test_FindDataItemByOneDimensionalPoint(self):
        # rightPoint - покрывающая точка
        point = Point([-0.3, 0.78], ["a", "b"])
        rightdataItem = SearchDataItem(point, 1, None, 1)
        rightdataItem.globalR = -3.2
        rightdataItem.localR = 1.03
        # leftPoint
        point = Point([-0.3, 0.78], ["e", "f"])
        leftdataItem = SearchDataItem(point, 0, None, 1)
        leftdataItem.globalR = 4.76
        leftdataItem.localR = 3.2

        self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)

        try:
            findRightDataItem = self.searchData.FindDataItemByOneDimensionalPoint(0.2)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

        self.assertEqual(findRightDataItem.GetX(), rightdataItem.GetX())
        self.assertEqual(findRightDataItem.GetY().floatVariables, rightdataItem.GetY().floatVariables)
        self.assertEqual(findRightDataItem.GetY().discreteVariables, rightdataItem.GetY().discreteVariables)
        self.assertEqual(findRightDataItem.globalR, rightdataItem.globalR)
        self.assertEqual(findRightDataItem.localR, rightdataItem.localR)

    def test_InsertDataItemWithoutRight(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, None, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemSecond = SearchDataItem(point, 1, None, 1)
        dataItemSecond.globalR = -3.2
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)

        point = Point([-0.4, 0.078], ["e", "f"])
        dataItem = SearchDataItem(point, 0.2, None, 1)
        dataItem.globalR = -4.76
        dataItem.localR = 0.2

        try:
            self.searchData.InsertDataItem(dataItem)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithEqulsCharactiristic(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, None, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.5, 0.02], ["e", "c"])
        dataItemSecond = SearchDataItem(point, 1, None, 1)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 1.03

        point = Point([0.13, -0.2], ["e", "c"])
        dataItemThird = SearchDataItem(point, 0.5, None, 1)
        dataItemThird.globalR = -0.476
        dataItemThird.localR = 3.2

        try:
            self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)
            self.searchData.InsertDataItem(dataItemThird)
        except Exception as exc:
            assert False, f"'self.searchData.InsertFirstDataItem'," \
                          f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_RefillQueue(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, None, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.5, 0.02], ["e", "c"])
        dataItemSecond = SearchDataItem(point, 1,None,  1)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 1.03

        self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)

        try:
            self.searchData.RefillQueue()
        except Exception as exc:
            assert False, f"'self.searchData.RefillQueue' raised an exception{exc}"

    def test_GetDataItemWithMaxGlobalR(self):
        point1 = Point([-0.6, 0.7], ["a", "f"])
        dataItem1 = SearchDataItem(point1, 0.05, None, 2)
        dataItem1.globalR = 4.10
        dataItem1.localR = 1.2

        point3 = Point([1.4, 3.7], ["a", "f"])
        dataItem3 = SearchDataItem(point3, 0.8, None, 1)
        dataItem3.globalR = 2.6
        dataItem3.localR = -0.8

        point2 = Point([-0.3, 0.78], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.2, None, 1)
        dataItem2.globalR = 5.0
        dataItem2.localR = 3.2

        self.searchData.InsertFirstDataItem(dataItem1, dataItem3)
        self.searchData.InsertDataItem(dataItem2)

        getDataItem = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 0.2)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['e', 'f'])

    def test_GetDataItemWithMaxLocalR(self):
        point1 = Point([-0.6, 0.7], ["a", "f"])
        dataItem1 = SearchDataItem(point1, 0.05, None, 2)
        dataItem1.globalR = 4.10
        dataItem1.localR = 1.2

        point3 = Point([1.4, 3.7], ["a", "f"])
        dataItem3 = SearchDataItem(point3, 0.8,  None, 1)
        dataItem3.globalR = 2.6
        dataItem3.localR = -0.8

        point2 = Point([-0.3, 0.78], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.2, None, 1)
        dataItem2.globalR = 5.0
        dataItem2.localR = 3.2

        self.searchData.InsertFirstDataItem(dataItem1, dataItem3)
        self.searchData.InsertDataItem(dataItem2)

        getDataItem = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 0.2)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.localR, 3.2)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['e', 'f'])

    def test_GetDataItemWithMaxGlobalRWithEaqualR(self):
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0, None, 1)
        dataItem1.globalR = 4.89
        dataItem1.localR = 3.2

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 1, None, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = 1.2

        dataItem2.SetLeft(dataItem1)

        point3 = Point([0.06, 0.17], ["c", "f"])
        dataItem3 = SearchDataItem(point3, 0.078, None, 1)
        dataItem3.globalR = 5.0
        dataItem3.localR = 0.2

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3)

        getDataItem = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 1)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.6, 0.7])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['a', 'f'])

    def test_GetDataItemWithMaxLocalRWithEqualR(self):
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0, None, 1)
        dataItem1.globalR = 4.89
        dataItem1.localR = 1.2

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 1, None, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = 1.2

        dataItem2.SetLeft(dataItem1)

        point3 = Point([0.06, 0.17], ["c", "f"])
        dataItem3 = SearchDataItem(point3, 0.078, None, 1)
        dataItem3.globalR = 5.0
        dataItem3.localR = 0.2

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3)

        getDataItem = self.searchData.GetDataItemWithMaxLocalR()

        self.assertEqual(getDataItem.GetX(), 0.0)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.localR, 1.2)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['e', 'f'])

    def test_CheckSearchData(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None)
        dataItem2 = SearchDataItem(([1.3, 4.7], ["a", "b"]), 1.0,  None)

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)

        dataItem3 = SearchDataItem(([2.1, -0.7], ["b", "f"]), 0.5,  None)
        dataItem3.globalR = 3.4
        self.searchData.InsertDataItem(dataItem3, dataItem2)

        getItem = self.searchData.GetDataItemWithMaxGlobalR()
        self.assertEqual(getItem.GetX(), 0.5)

        dataItem3.globalR = 2.7
        dataItem4 = SearchDataItem(([3.06, 1.67], ["d", "f"]), 0.2,  None)
        dataItem4.globalR = 1.2
        self.searchData.InsertDataItem(dataItem4, dataItem3)

        getItem = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getItem.GetX(), 0.5)

    def test_CheckQueueLinked(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None)
        dataItem1.globalR = 4.10
        dataItem1.localR = 1.2

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None)
        dataItem2.globalR = 5.0
        dataItem2.localR = 4.56

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None)
        dataItem3.globalR = 2.6
        dataItem3.localR = -0.8

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3)

        getDataItemGlob = self.searchData.GetDataItemWithMaxGlobalR()
        getDataItemGlob2 = self.searchData.GetDataItemWithMaxGlobalR()
        getDataItemLocal = self.searchData.GetDataItemWithMaxLocalR()

        self.assertEqual(getDataItemGlob.GetX(), 1.0)
        self.assertEqual(getDataItemGlob.globalR, 5.0)

        self.assertEqual(getDataItemGlob2.GetX(), 0.0)
        self.assertEqual(getDataItemGlob2.globalR, 4.10)

        self.assertEqual(getDataItemLocal.GetX(), 0.8)
        self.assertEqual(getDataItemLocal.localR, -0.8)

    def test_GetCount(self):
        point = Point([-0.01, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1, None)
        dataItemFirst.globalR = 0.76
        dataItemFirst.localR = 3.2

        point = Point([-0.5, 0.02], ["e", "c"])
        dataItemSecond = SearchDataItem(point, 1, 1, None)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 1.03

        self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)

        self.assertEqual(self.searchData.GetCount(), 2)


# Executing the tests in the above test case class


if __name__ == "__main__":
    unittest.main()
