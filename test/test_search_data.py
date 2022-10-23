import unittest

import iOpt.method.search_data
import iOpt.trial

from iOpt.method.search_data import SearchDataItem
from iOpt.trial import Point


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
        self.searchDataItem.SetIndex(-1)
        self.assertEqual(self.searchDataItem.GetIndex(), -1)

    def test_SetZ(self):
        self.searchDataItem.SetZ(-4.0)
        self.assertEqual(self.searchDataItem.GetZ(), -4.0)

    def test_SetLeft(self):
        pointl = Point([0.3, 0.78], ["c", "d"])
        leftDataItem = SearchDataItem(pointl, 0.6, 1)
        self.searchDataItem.SetLeft(leftDataItem)

        self.assertEqual(self.searchDataItem.GetLeft().GetX(), 0.6)
        self.assertEqual(self.searchDataItem.GetLeft().GetDiscreteValueIndex(), 1)
        self.assertEqual(self.searchDataItem.GetLeft().GetY().floatVariables, [0.3, 0.78])
        self.assertEqual(self.searchDataItem.GetLeft().GetY().discreteVariables, ['c', 'd'])

    def test_SetRigth(self):
        pointr = Point([-0.3, 0.78], ["e", "f"])
        rightDataItem = SearchDataItem(pointr, 0.16, 0)
        self.searchDataItem.SetRigth(rightDataItem)

        self.assertEqual(self.searchDataItem.GetRigth().GetX(), 0.16)
        self.assertEqual(self.searchDataItem.GetRigth().GetDiscreteValueIndex(), 0)
        self.assertEqual(self.searchDataItem.GetRigth().GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(self.searchDataItem.GetRigth().GetY().discreteVariables, ['e', 'f'])


class TestCharacteristicsQueue(unittest.TestCase):
    def setUp(self):
        self.characteristicsQueueGlobalR = search_data.CharacteristicsQueue(typeR=1)
        self.characteristicsQueueLocalR = search_data.CharacteristicsQueue(typeR=0)

    def test_Insert(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItem = SearchDataItem(point, 0.2, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2
        try:
            self.characteristicsQueueLocalR.Insert(dataItem)
            self.characteristicsQueueGlobalR.Insert(dataItem)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Insert'," \
                          f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_InsertWithEqualR(self):

        point = Point([-0.3, 0.78], ["e", "f"])
        dataItem = SearchDataItem(point, 0.2, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2

        point = Point([-0.5, 0.07], ["a", "f"])
        dataItem = SearchDataItem(point, 0.6, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2
        try:
            self.characteristicsQueueLocalR.Insert(dataItem)
            self.characteristicsQueueGlobalR.Insert(dataItem)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Insert'," \
                          f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_Clear(self):
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItem = SearchDataItem(point, 0.2, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2

        self.characteristicsQueueLocalR.Insert(dataItem)
        self.characteristicsQueueGlobalR.Insert(dataItem)

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
        dataItem1 = SearchDataItem(point1, 0.2, 1)
        dataItem1.globalR = 4.76
        dataItem1.localR = 3.2
        self.characteristicsQueueGlobalR.Insert(dataItem1)

        point2 = Point([-0.6, 0.7], ["e", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, 1)
        dataItem2.globalR = 5.0
        dataItem2.localR = 3.8
        self.characteristicsQueueGlobalR.Insert(dataItem2)

        tupleDataItem = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItem = tupleDataItem[1]

        self.assertEqual(getDataItem.GetX(), 0.05)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.6, 0.7])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['e', 'f'])

    def test_CanNotGetBestItem(self):
        self.assertEqual(self.characteristicsQueueGlobalR.GetBestItem(), None)

    def test_CanGetBestItmeWithEqualR(self):
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0.2, 1)
        dataItem1.globalR = 5.0
        dataItem1.localR = 3.2
        self.characteristicsQueueGlobalR.Insert(dataItem1)
        self.characteristicsQueueLocalR.Insert(dataItem1)

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = 1.2
        self.characteristicsQueueGlobalR.Insert(dataItem2)
        self.characteristicsQueueLocalR.Insert(dataItem2)

        point2 = Point([0.06, 0.17], ["c", "f"])
        dataItem2 = SearchDataItem(point2, 0.078, 1)
        dataItem2.globalR = -4.0
        dataItem2.localR = 3.2
        self.characteristicsQueueGlobalR.Insert(dataItem2)
        self.characteristicsQueueLocalR.Insert(dataItem2)

        tupleDataItemG = self.characteristicsQueueGlobalR.GetBestItem()
        getDataItemG = tupleDataItemG[1]

        tupleDataItemL = self.characteristicsQueueLocalR.GetBestItem()
        getDataItemL = tupleDataItemL[1]

        self.assertEqual(getDataItemG.GetX(), 0.05)
        self.assertEqual(getDataItemG.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItemG.globalR, 5.0)
        self.assertEqual(getDataItemG.GetY().floatVariables, [-0.6, 0.7])
        self.assertEqual(getDataItemG.GetY().discreteVariables, ['a', 'f'])

        self.assertEqual(getDataItemL.GetX(), 0.078)
        self.assertEqual(getDataItemL.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItemL.localR, 3.2)
        self.assertEqual(getDataItemL.GetY().floatVariables, [0.06, 0.17])
        self.assertEqual(getDataItemL.GetY().discreteVariables, ['c', 'f'])


class TestSearchData(unittest.TestCase):

    def test_ClearQueue(self):
        searchData = search_data.SearchData(None)
        try:
            searchData.ClearQueue()
        except Exception as exc:
            assert False, f"'self.searchData.ClearQueue' raised an exception{exc}"

    def test_InsertDataItemFirstIteration(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItem = SearchDataItem(point, 0, 1)
        dataItem.globalR = 4.76
        dataItem.localR = 3.2

        point = Point([-0.3, 0.78], ["e", "f"])
        rightdataItem = SearchDataItem(point, 1, 1)
        rightdataItem.globalR = -3.2
        rightdataItem.localR = 1.03

        rightdataItem.SetLeft(dataItem)
        try:
            searchData.InsertDataItem(dataItem)
            searchData.InsertDataItem(rightdataItem)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithRight(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemSecond = SearchDataItem(point, 1, 1)
        dataItemSecond.globalR = -3.2
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        searchData.InsertDataItem(dataItemFirst)
        searchData.InsertDataItem(dataItemSecond)

        point = Point([1.4, 3.078], ["e", "f"])
        dataItem = SearchDataItem(point, 0.5, 1)
        dataItem.globalR = -0.9
        dataItem.localR = 0.5

        try:
            searchData.InsertDataItem(dataItem, dataItemSecond)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_FindDataItemByOneDimensionalPoint(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemSecond = SearchDataItem(point, 1, 1)
        dataItemSecond.globalR = -3.2
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        searchData.InsertDataItem(dataItemFirst)
        searchData.InsertDataItem(dataItemSecond)

        point = Point([1.4, 3.078], ["e", "f"])
        dataItem = SearchDataItem(point, 0.5, 1)
        dataItem.globalR = -0.9
        dataItem.localR = 0.5

        try:
            findRightDataItem = searchData.FindDataItemByOneDimensionalPoint(dataItem.GetX())
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

        self.assertEqual(findRightDataItem.GetX(), dataItemSecond.GetX())
        self.assertEqual(findRightDataItem.GetY().floatVariables, dataItemSecond.GetY().floatVariables)
        self.assertEqual(findRightDataItem.GetY().discreteVariables, dataItemSecond.GetY().discreteVariables)
        self.assertEqual(findRightDataItem.globalR, dataItemSecond.globalR)
        self.assertEqual(findRightDataItem.localR, dataItemSecond.localR)

    def test_InsertDataItemWithoutRight(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemSecond = SearchDataItem(point, 1, 1)
        dataItemSecond.globalR = -3.2
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        searchData.InsertDataItem(dataItemFirst)
        searchData.InsertDataItem(dataItemSecond)

        point = Point([-0.4, 0.078], ["e", "f"])
        dataItem = SearchDataItem(point, 0.2, 1)
        dataItem.globalR = -4.76
        dataItem.localR = 0.2

        try:
            searchData.InsertDataItem(dataItem)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithEqulsCharactiristic(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.5, 0.02], ["e", "c"])
        dataItemSecond = SearchDataItem(point, 1, 1)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        point = Point([0.13, -0.2], ["e", "c"])
        dataItemThird = SearchDataItem(point, 0.5, 1)
        dataItemThird.globalR = -0.476
        dataItemThird.localR = 3.2

        try:
            searchData.InsertDataItem(dataItemFirst)
            searchData.InsertDataItem(dataItemSecond)
            searchData.InsertDataItem(dataItemThird)
        except Exception as exc:
            assert False, f"'self.searchData.InsertDataItem', raised an exception{exc}"

    def test_RefillQueue(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.3, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1)
        dataItemFirst.globalR = 4.76
        dataItemFirst.localR = 3.2

        point = Point([-0.5, 0.02], ["e", "c"])
        dataItemSecond = SearchDataItem(point, 1, 1)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        searchData.InsertDataItem(dataItemFirst)
        searchData.InsertDataItem(dataItemSecond)

        try:
            searchData.RefillQueue()
        except Exception as exc:
            assert False, f"'self.searchData.RefillQueue' raised an exception{exc}"

    def test_GetDataItemWithMaxGlobalR(self):
        searchData = search_data.SearchData(None)
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0.2, 1)
        dataItem1.globalR = 5.0
        dataItem1.localR = 3.2

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, 2)
        dataItem2.globalR = 4.10
        dataItem2.localR = 1.2

        dataItem1.SetLeft(dataItem2)

        searchData.InsertDataItem(dataItem2)
        searchData.InsertDataItem(dataItem1)

        tupleDataItemG = searchData.GetDataItemWithMaxGlobalR()
        getDataItemG = tupleDataItemG[1]

        self.assertEqual(getDataItemG.GetX(), 0.2)
        self.assertEqual(getDataItemG.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItemG.globalR, 5.0)
        self.assertEqual(getDataItemG.GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(getDataItemG.GetY().discreteVariables, ['e', 'f'])


    def test_GetDataItemWithMaxLocalR(self):
        searchData = search_data.SearchData(None)
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0.2, 1)
        dataItem1.globalR = -5.0
        dataItem1.localR = 3.2

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, 2)
        dataItem2.globalR = 4.10
        dataItem2.localR = 1.2

        dataItem1.SetLeft(dataItem2)

        searchData.InsertDataItem(dataItem2)
        searchData.InsertDataItem(dataItem1)

        tupleDataItemG = searchData.GetDataItemWithMaxLocalR()
        getDataItemG = tupleDataItemG[1]

        self.assertEqual(getDataItemG.GetX(), 0.2)
        self.assertEqual(getDataItemG.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItemG.localR, 3.2)
        self.assertEqual(getDataItemG.GetY().floatVariables, [-0.3, 0.78])
        self.assertEqual(getDataItemG.GetY().discreteVariables, ['e', 'f'])

    def test_GetDataItemWithMaxGlobalRWithEaqualR(self):
        searchData = search_data.SearchData(None)
        point1 = Point([-0.3, 0.78], ["e", "f"])
        dataItem1 = SearchDataItem(point1, 0, 1)
        dataItem1.globalR = 4.89
        dataItem1.localR = 3.2

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 1, 2)
        dataItem2.globalR = 5.0
        dataItem2.localR = 1.2

        dataItem2.SetLeft(dataItem1)

        point3 = Point([0.06, 0.17], ["c", "f"])
        dataItem3 = SearchDataItem(point3, 0.078, 1)
        dataItem3.globalR = 5.0
        dataItem3.localR = 0.2

        searchData.InsertDataItem(dataItem1)
        searchData.InsertDataItem(dataItem2)
        searchData.InsertDataItem(dataItem3)

        tupleDataItemG = searchData.GetDataItemWithMaxGlobalR()
        getDataItemG = tupleDataItemG[1]

        self.assertEqual(getDataItemG.GetX(), 0.078)
        self.assertEqual(getDataItemG.GetDiscreteValueIndex(), 1)
        self.assertEqual(getDataItemG.globalR, 5.0)
        self.assertEqual(getDataItemG.GetY().floatVariables, [0.06, 0.17])
        self.assertEqual(getDataItemG.GetY().discreteVariables, ['c', 'f'])

    def test_GetDataItemWithMaxLocalRWithEqualR(self):
        searchData = search_data.SearchData(None)
        point1 = Point([0.73, 0.07], ["a", "b"])
        dataItem1 = SearchDataItem(point1, 0.02, 1)
        dataItem1.globalR = -0.03
        dataItem1.localR = -0.2

        point2 = Point([-0.6, 0.7], ["a", "f"])
        dataItem2 = SearchDataItem(point2, 0.05, 2)
        dataItem2.globalR = 4.10
        dataItem2.localR = 1.2

        dataItem1.SetLeft(dataItem2)

        point3 = Point([0.06, 0.17], ["c", "f"])
        dataItem3 = SearchDataItem(point3, 0.78, 1)
        dataItem3.globalR = 5.0
        dataItem3.localR = 1.2

        searchData.InsertDataItem(dataItem2)
        searchData.InsertDataItem(dataItem1)
        searchData.InsertDataItem(dataItem3)

        tupleDataItem = searchData.GetDataItemWithMaxLocalR()
        getDataItem = tupleDataItem[1]

        self.assertEqual(getDataItem.GetX(), 0.05)
        self.assertEqual(getDataItem.GetDiscreteValueIndex(), 2)
        self.assertEqual(getDataItem.localR, 1.2)
        self.assertEqual(getDataItem.GetY().floatVariables, [-0.6, 0.7])
        self.assertEqual(getDataItem.GetY().discreteVariables, ['a', 'f'])


    def test_GetCount(self):
        searchData = search_data.SearchData(None)
        point = Point([-0.01, 0.78], ["e", "f"])
        dataItemFirst = SearchDataItem(point, 0, 1)
        dataItemFirst.globalR = 0.76
        dataItemFirst.localR = 3.2

        point = Point([-0.5, 0.02], ["e", "c"])
        dataItemSecond = SearchDataItem(point, 1, 1)
        dataItemSecond.globalR = 4.76
        dataItemSecond.localR = 1.03

        dataItemSecond.SetLeft(dataItemFirst)

        searchData.InsertDataItem(dataItemFirst)
        searchData.InsertDataItem(dataItemSecond)

        self.assertEqual(searchData.GetCount(), 2)

# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()