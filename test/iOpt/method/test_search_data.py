import unittest

import iOpt.method.search_data
import iOpt.trial

from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point

from iOpt.method.search_data import CharacteristicsQueue
from iOpt.method.search_data import SearchData
from iOpt.method.search_data import SearchDataDualQueue


class TestSearchDataItem(unittest.TestCase):
    # setUp method is overridden from the parent class SearchDataItem
    def setUp(self):
        self.point = Point([0.1, 0.5], ["a", "b"])
        self.searchDataItem = SearchDataItem(self.point, 0.3, None, 0)

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
        leftDataItem = SearchDataItem(([0.3, 0.78], ["c", "d"]), 0.6, None, 1)
        self.searchDataItem.SetLeft(leftDataItem)

        self.assertEqual(self.searchDataItem.GetLeft().GetX(), 0.6)
        self.assertEqual(self.searchDataItem.GetLeft().GetDiscreteValueIndex(), 1)
        self.assertEqual(self.searchDataItem.GetLeft().GetY(), ([0.3, 0.78], ['c', 'd']))

    def test_SetRigth(self):
        rightDataItem = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.16, None, 0)
        self.searchDataItem.SetRight(rightDataItem)

        self.assertEqual(self.searchDataItem.GetRight().GetX(), 0.16)
        self.assertEqual(self.searchDataItem.GetRight().GetDiscreteValueIndex(), 0)
        self.assertEqual(self.searchDataItem.GetRight().GetY(), ([-0.3, 0.78], ['e', 'f']))


class TestCharacteristicsQueue(unittest.TestCase):
    def setUp(self):
        self.characteristicsQueueGlobalR = CharacteristicsQueue(maxlen=3)

    def test_Insert(self):
        dataItem = SearchDataItem(([1.03, 0.5], ["k", "m"]), 0, None, 1)

        try:
            self.characteristicsQueueGlobalR.Insert(dataItem.globalR, dataItem)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Insert'," \
                          f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_InsertWithEqualR(self):
        dataItem = SearchDataItem(([-0.3, 0.78], ["e", "m"]), 0.2, None, 1)
        dataItem.globalR = 4.76

        dataItem = SearchDataItem(([-0.5, 0.07], ["a", "f"]), 0.6, None, 1)
        dataItem.globalR = 4.76
        try:
            self.characteristicsQueueGlobalR.Insert(dataItem.globalR, dataItem)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_Clear(self):
        try:
            self.characteristicsQueueGlobalR.Clear()
        except Exception as exc:
            assert False, f"'self.characteristicsQueueGlobalR.Clear' raised an exception{exc}"
        self.assertEqual(self.characteristicsQueueGlobalR.IsEmpty(), True)

    def test_CanGetBestItem(self):
        dataItem1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, None, 1)
        dataItem1.globalR = 4.76

        dataItem2 = SearchDataItem(([-0.6, 0.7], ["e", "f"]), 0.05, None, 2)
        dataItem2.globalR = 5.0

        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)
        getDataItem = self.characteristicsQueueGlobalR.GetBestItem()[0]

        self.assertEqual(getDataItem.GetX(), 0.05)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY(), ([-0.6, 0.7], ["e", "f"]))

    def test_CanGetBestItmeWithEqualGlobalR(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.05, None, 2)
        dataItem1.globalR = 5.0

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.4, None, 1)
        dataItem2.globalR = 5.0

        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)
        getDataItemG = self.characteristicsQueueGlobalR.GetBestItem()[0]

        self.assertEqual(getDataItemG.GetX(), 0.05)
        self.assertEqual(getDataItemG.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItemG.globalR, 5.0)
        self.assertEqual(getDataItemG.GetY(), ([-0.6, 0.7], ['a', 'f']))

    def test_GetLen(self):
        dataItem1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, 1)

        dataItem2 = SearchDataItem(([-0.6, 0.7], ["e", "f"]), 0.05, 2)

        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(), 2)

    def test_GetMaxLen(self):
        self.assertEqual(self.characteristicsQueueGlobalR.GetMaxLen(), 3)

    def test_NotAddItemWithLowerPriority(self):
        dataItem1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, None, 1)
        dataItem1.globalR = 4.76

        dataItem2 = SearchDataItem(([-0.6, 0.7], ["s", "k"]), 0.05, None, 2)
        dataItem2.globalR = 5.0

        dataItem3 = SearchDataItem(([1.08, 4.56], ["a", "b"]), 0.7, None, 1)
        dataItem3.globalR = 3.05

        dataItem4 = SearchDataItem(([0.076, 2.7], ["c", "d"]), 0.5, None, 1)
        dataItem4.globalR = 1.5

        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)
        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)
        self.characteristicsQueueGlobalR.Insert(dataItem3.globalR, dataItem3)
        self.characteristicsQueueGlobalR.Insert(dataItem4.globalR, dataItem4)

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(),
                         self.characteristicsQueueGlobalR.GetMaxLen())

        getDataItem1 = self.characteristicsQueueGlobalR.GetBestItem()[0]
        getDataItem2 = self.characteristicsQueueGlobalR.GetBestItem()[0]
        getDataItem3 = self.characteristicsQueueGlobalR.GetBestItem()[0]

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(), 0)

        self.assertEqual(getDataItem1.GetX(), 0.05)
        self.assertEqual(getDataItem1.globalR, 5.0)

        self.assertEqual(getDataItem2.GetX(), 0.2)
        self.assertEqual(getDataItem2.globalR, 4.76)

        self.assertEqual(getDataItem3.GetX(), 0.7)
        self.assertEqual(getDataItem3.globalR, 3.05)

    def test_AddItemWithHightPriority(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.05, 2)
        dataItem1.globalR = 4.10
        self.characteristicsQueueGlobalR.Insert(dataItem1.globalR, dataItem1)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.74, 1)
        dataItem2.globalR = 2.50

        dataItem3 = SearchDataItem(([1.5, 3.2], ["r", "s"]), 0.12, 0)
        dataItem3.globalR = 3.23

        dataItem4 = SearchDataItem(([2.67, -0.78], ["h", "k"]), 0.4, 1)
        dataItem4.globalR = 3.0

        self.characteristicsQueueGlobalR.Insert(dataItem2.globalR, dataItem2)
        self.characteristicsQueueGlobalR.Insert(dataItem3.globalR, dataItem3)
        self.characteristicsQueueGlobalR.Insert(dataItem4.globalR, dataItem4)

        self.assertEqual(self.characteristicsQueueGlobalR.GetLen(),
                         self.characteristicsQueueGlobalR.GetMaxLen())
        getDataItem1 = self.characteristicsQueueGlobalR.GetBestItem()[0]
        getDataItem2 = self.characteristicsQueueGlobalR.GetBestItem()[0]
        getDataItem3 = self.characteristicsQueueGlobalR.GetBestItem()[0]

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
        leftdataItem = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        rightdataItem = SearchDataItem(([1.3, -1.78], ["e", "k"]), 1, None, 1)
        rightdataItem.globalR = -3.2

        try:
            self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)
        except Exception as exc:
            assert False, f"' self.searchData.InsertDataItemFirst' raised an exception{exc}"

    def test_InsertDataItemWithRight(self):
        leftdataItem = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItem = SearchDataItem(([1.4, 3.078], ["c", "f"]), 0.5, None, 1)
        dataItem.globalR = -0.9

        rightdataItem = SearchDataItem(([-0.3, 0.78], ["a", "b"]), 1, None, 1)
        rightdataItem.globalR = -3.2

        self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)

        try:
            self.searchData.InsertDataItem(dataItem, rightdataItem)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_CanUseIteration(self):
        leftdataItem = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItem = SearchDataItem(([1.2, 2.078], ["l", "f"]), 0.5, None, 1)

        rightdataItem = SearchDataItem(([0.93, -2.0], ["e", "a"]), 1, None, 1)

        self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)
        self.searchData.InsertDataItem(dataItem, rightdataItem)

        getData = []
        try:
            for item in self.searchData:
                getData.append(item.GetX())
        except Exception as exc:
            assert False, f"'item.GetX' raised an exception{exc}"

        self.assertEqual(self.searchData.GetCount(), 3)
        self.assertEqual(getData[0], leftdataItem.GetX())
        self.assertEqual(getData[1], dataItem.GetX())
        self.assertEqual(getData[2], rightdataItem.GetX())

    def test_FindDataItemByOneDimensionalPoint(self):
        leftdataItem = SearchDataItem(([1.6, -2.78], ["e", "l"]), 0, None, 1)

        rightdataItem = SearchDataItem(([-0.3, 0.78], ["a", "b"]), 1, None, 1)
        rightdataItem.globalR = 1.0

        self.searchData.InsertFirstDataItem(leftdataItem, rightdataItem)

        try:
            findRightDataItem = self.searchData.FindDataItemByOneDimensionalPoint(0.2)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

        self.assertEqual(findRightDataItem.GetX(), rightdataItem.GetX())
        self.assertEqual(findRightDataItem.GetY(), rightdataItem.GetY())
        self.assertEqual(findRightDataItem.globalR, rightdataItem.globalR)

    def test_InsertDataItemWithoutRight(self):
        dataItem1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItem2 = SearchDataItem(([0.0, 1.28], ["n", "f"]), 1, None, 1)
        dataItem2.globalR = 1.0

        dataItem3 = SearchDataItem(([-0.4, 0.078], ["e", "f"]), 0.2, None, 1)
        dataItem3.globalR = -4.76

        dataItem4 = SearchDataItem(([2.09, 1.15], ["l", "l"]), 0.16, None, 1)
        dataItem4.globalR = -3.6

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3)
        try:
            self.searchData.InsertDataItem(dataItem4)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithEqualCharactiristics(self):
        dataItemFirst = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItemSecond = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        dataItemSecond.globalR = 4.76

        dataItemThird = SearchDataItem(([0.13, -0.2], ["e", "c"]), 0.5, None, 1)
        dataItemThird.globalR = 4.76

        self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)
        try:
            self.searchData.InsertDataItem(dataItemThird, dataItemSecond)
        except Exception as exc:
            assert False, f"'self.searchData.InsertFirstDataItem'," \
                          f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_RefillQueue(self):
        dataItemFirst = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItemSecond = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        dataItemSecond.globalR = 1.0

        self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)

        try:
            self.searchData.RefillQueue()
        except Exception as exc:
            assert False, f"'self.searchData.RefillQueue' raised an exception{exc}"

    def test_GetDataItemWithMaxGlobalR(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.05, None, 2)
        dataItem1.globalR = 4.10

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, None, 1)
        dataItem2.globalR = 5.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.globalR = 2.6

        self.searchData.InsertFirstDataItem(dataItem1, dataItem3)
        self.searchData.InsertDataItem(dataItem2)

        getDataItem = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 0.2)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY(), ([-0.3, 0.78], ['e', 'f']))

    def test_GetDataItemWithMaxGlobalRWithEaqualR(self):
        dataItem1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)
        dataItem2 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 1, None, 2)
        dataItem2.globalR = 1.0
        dataItem3 = SearchDataItem(([0.06, 0.17], ["c", "f"]), 0.078, None, 2)
        dataItem3.globalR = 5.0
        dataItem4 = SearchDataItem(([2.15, -4.17], ["b", "k"]), 0.7, None, 1)
        dataItem4.globalR = 5.0

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3, dataItem2)
        self.searchData.InsertDataItem(dataItem4)

        getDataItem = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 0.078)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY(), ([0.06, 0.17], ['c', 'f']))

    def test_Check(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.globalR = 1.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.globalR = 2.6

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3, dataItem2)

        getDataItemGlob = self.searchData.GetDataItemWithMaxGlobalR()
        getDataItemGlob2 = self.searchData.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItemGlob.GetX(), 0.8)
        self.assertEqual(getDataItemGlob.globalR, 2.6)

        self.assertEqual(getDataItemGlob2.GetX(), 1.0)
        self.assertEqual(getDataItemGlob2.globalR, 1.0)

    def test_GetCount(self):
        dataItemFirst = SearchDataItem(([-0.01, 0.78], ["e", "f"]), 0, 1)

        dataItemSecond = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, 1)

        self.searchData.InsertFirstDataItem(dataItemFirst, dataItemSecond)

        self.assertEqual(self.searchData.GetCount(), 2)

    def test_GetLastItem(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.globalR = 1.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.globalR = 2.6

        self.searchData.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchData.InsertDataItem(dataItem3, dataItem2)

        getLastItem = self.searchData.GetLastItem()
        self.assertEqual(getLastItem.GetX(), dataItem3.GetX())

    def test_GetLastItemExcept(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.globalR = 1.0
        try:
            getLastItem = self.searchData.GetLastItem()
        except Exception as exc:
            assert True, f"'self.searchData.GetLastItem()' raised an exception{exc}"


class TestSearchDataDualQueue(unittest.TestCase):
    def setUp(self):
        self.searchDataDual = SearchDataDualQueue(None, maxlen=3)

    def test_ClearQueue(self):
        try:
            self.searchDataDual.ClearQueue()
        except Exception as exc:
            assert False, f"'self.searchDataDual.ClearQueue' raised an exception{exc}"

    def test_InsertDataItemWithRight(self):
        leftdataItem = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItem = SearchDataItem(([1.4, 3.078], ["c", "f"]), 0.5, None, 1)
        dataItem.globalR = -0.9
        dataItem.localR = 1.2

        rightdataItem = SearchDataItem(([-0.3, 0.78], ["a", "b"]), 1, None, 1)
        rightdataItem.globalR = -3.2
        rightdataItem.localR = 2.03

        self.searchDataDual.InsertFirstDataItem(leftdataItem, rightdataItem)

        try:
            self.searchDataDual.InsertDataItem(dataItem, rightdataItem)
        except Exception as exc:
            assert False, f"'self.searchDataDual.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithoutRight(self):
        dataItem1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItem2 = SearchDataItem(([0.0, 1.28], ["n", "f"]), 1, None, 1)
        dataItem2.globalR = 1.0
        dataItem2.localR = 1.0

        dataItem3 = SearchDataItem(([-0.4, 0.078], ["e", "f"]), 0.2, None, 1)
        dataItem3.globalR = -4.76
        dataItem3.localR = 1.34

        point = Point([2.09, 1.15], ["l", "l"])
        dataItem4 = SearchDataItem(point, 0.16, None, 1)
        dataItem4.globalR = 3.6
        dataItem4.localR = -0.67

        self.searchDataDual.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem3)
        try:
            self.searchDataDual.InsertDataItem(dataItem4)
        except Exception as exc:
            assert False, f"'self.searchDataDual.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithEqualCharactiristics(self):
        dataItemFirst = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItemSecond = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 0.67

        point = Point([0.13, -0.2], ["e", "c"])
        dataItemThird = SearchDataItem(point, 0.5, None, 1)
        dataItemThird.globalR = 4.76
        dataItemThird.localR = 0.67

        self.searchDataDual.InsertFirstDataItem(dataItemFirst, dataItemSecond)
        try:
            self.searchDataDual.InsertDataItem(dataItemThird, dataItemSecond)
        except Exception as exc:
            assert False, f"'self.searchDataDual.InsertFirstDataItem'," \
                          f"'self.searchDataDual.InsertDataItem' raised an exception{exc}"

    def test_RefillQueue(self):
        dataItemFirst = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        dataItemSecond = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        dataItemSecond.globalR = 1.0
        dataItemSecond.localR = 1.0

        self.searchDataDual.InsertFirstDataItem(dataItemFirst, dataItemSecond)

        try:
            self.searchDataDual.RefillQueue()
        except Exception as exc:
            assert False, f"'self.searchDataDual.RefillQueue' raised an exception{exc}"

    def test_GetDataItemWithMaxGlobalR(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.globalR = -5.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.globalR = 3.6

        dataItem4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        dataItem4.globalR = 2.8

        self.searchDataDual.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem3, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem4)

        getDataItem = self.searchDataDual.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 0.8)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.globalR, 3.6)
        self.assertEqual(getDataItem.GetY(), ([1.4, 3.7], ["a", "f"]))

    def test_GetDataItemWithMaxLocalR(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.localR = -5.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.localR = 3.6

        dataItem4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        dataItem4.localR = 2.8

        self.searchDataDual.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem3, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem4)

        getDataItem = self.searchDataDual.GetDataItemWithMaxLocalR()

        self.assertEqual(getDataItem.GetX(), 0.8)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.localR, 3.6)
        self.assertEqual(getDataItem.GetY(), ([1.4, 3.7], ["a", "f"]))

    def test_GetDataItemWithMaxGlobalRWithEqualCharacteristic(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.globalR = -5.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.globalR = 3.6

        dataItem4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        dataItem4.globalR = 3.6

        self.searchDataDual.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem3, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem4)

        getDataItem = self.searchDataDual.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItem.GetX(), 0.8)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.globalR, 3.6)
        self.assertEqual(getDataItem.GetY(), ([1.4, 3.7], ["a", "f"]))

    def test_GetDataItemWithMaxLocalRWithEqualCharacteristic(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.localR = -5.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        dataItem3.localR = 3.6

        dataItem4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        dataItem4.localR = 3.6

        self.searchDataDual.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem3, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem4)

        getDataItem = self.searchDataDual.GetDataItemWithMaxLocalR()

        self.assertEqual(getDataItem.GetX(), 0.8)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.localR, 3.6)
        self.assertEqual(getDataItem.GetY(), ([1.4, 3.7], ["a", "f"]))

    def test_CheckQueueLinked(self):
        dataItem1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        dataItem2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        dataItem2.globalR = 1.0
        dataItem2.localR = 1.0

        dataItem3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.5, None, 1)
        dataItem3.globalR = 2.6
        dataItem3.localR = -1.09

        dataItem4 = SearchDataItem(([-0.7, 3.1], ["s", "w"]), 0.25, None, 2)
        dataItem4.globalR = 0.12
        dataItem4.localR = -2.4

        dataItem5 = SearchDataItem(([9.7, -0.1], ["m", "b"]), 0.76, None, 2)
        dataItem5.globalR = -0.134
        dataItem5.localR = 0.082

        self.searchDataDual.InsertFirstDataItem(dataItem1, dataItem2)
        self.searchDataDual.InsertDataItem(dataItem3, dataItem2)
        getDataItemGlob = self.searchDataDual.GetDataItemWithMaxGlobalR()

        self.assertEqual(getDataItemGlob.GetX(), 0.5)
        self.assertEqual(getDataItemGlob.globalR, 2.6)

        getDataItemGlob.globalR = -1.89
        getDataItemGlob.localR = -1.37
        self.searchDataDual.InsertDataItem(dataItem4, getDataItemGlob)

        getDataItemGlob2 = self.searchDataDual.GetDataItemWithMaxGlobalR()
        self.assertEqual(getDataItemGlob2.GetX(), 1.0)
        self.assertEqual(getDataItemGlob2.globalR, 1.0)

        getDataItemGlob2.globalR = 0.5
        getDataItemGlob2.localR = 0.5
        self.searchDataDual.InsertDataItem(dataItem5, getDataItemGlob2)

        getDataItemLocal = self.searchDataDual.GetDataItemWithMaxLocalR()
        self.assertEqual(getDataItemLocal.GetX(), 1.0)
        self.assertEqual(getDataItemLocal.localR, 0.5)


# Executing the tests in the above test case class


if __name__ == "__main__":
    unittest.main()
