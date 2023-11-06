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
        self.search_dataItem = SearchDataItem(self.point, 0.3, None, 0)

    def test_Init(self):
        self.assertEqual(self.search_dataItem.point.float_variables, [0.1, 0.5])

    def test_GetX(self):
        self.assertEqual(self.search_dataItem.get_x(), 0.3)

    def test_GetY(self):
        self.assertEqual((self.search_dataItem.get_y().float_variables,
                          self.search_dataItem.get_y().discrete_variables),
                         ([0.1, 0.5], ['a', 'b']))

    def test_Getdiscrete_value_index(self):
        self.assertEqual(self.search_dataItem.get_discrete_value_index(), 0)

    def test_SetIndex(self):
        self.search_dataItem.set_index(1)
        self.assertEqual(self.search_dataItem.get_index(), 1)

    def test_SetZ(self):
        self.search_dataItem.set_z(-4.0)
        self.assertEqual(self.search_dataItem.get_z(), -4.0)

    def test_SetLeft(self):
        leftDataItem = SearchDataItem(([0.3, 0.78], ["c", "d"]), 0.6, None, 1)
        self.search_dataItem.set_left(leftDataItem)

        self.assertEqual(self.search_dataItem.get_left().get_x(), 0.6)
        self.assertEqual(self.search_dataItem.get_left().get_discrete_value_index(), 1)
        self.assertEqual(self.search_dataItem.get_left().get_y(), ([0.3, 0.78], ['c', 'd']))

    def test_SetRigth(self):
        rightDataItem = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.16, None, 0)
        self.search_dataItem.set_right(rightDataItem)

        self.assertEqual(self.search_dataItem.get_right().get_x(), 0.16)
        self.assertEqual(self.search_dataItem.get_right().get_discrete_value_index(), 0)
        self.assertEqual(self.search_dataItem.get_right().get_y(), ([-0.3, 0.78], ['e', 'f']))


class TestCharacteristicsQueue(unittest.TestCase):
    def setUp(self):
        self.characteristicsQueueGlobalR = CharacteristicsQueue(maxlen=3)

    def test_Insert(self):
        data_item = SearchDataItem(([1.03, 0.5], ["k", "m"]), 0, None, 1)

        try:
            self.characteristicsQueueGlobalR.insert(data_item.globalR, data_item)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueLocalR.Insert'," \
                          f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_InsertWithEqualR(self):
        data_item = SearchDataItem(([-0.3, 0.78], ["e", "m"]), 0.2, None, 1)
        data_item.globalR = 4.76

        data_item = SearchDataItem(([-0.5, 0.07], ["a", "f"]), 0.6, None, 1)
        data_item.globalR = 4.76
        try:
            self.characteristicsQueueGlobalR.insert(data_item.globalR, data_item)
        except Exception as exc:
            assert False, f"'self.characteristicsQueueGlobalR.Insert' raised an exception{exc}"

    def test_Clear(self):
        try:
            self.characteristicsQueueGlobalR.Clear()
        except Exception as exc:
            assert False, f"'self.characteristicsQueueGlobalR.Clear' raised an exception{exc}"
        self.assertEqual(self.characteristicsQueueGlobalR.is_empty(), True)

    def test_CanGetBestItem(self):
        data_item1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, None, 1)
        data_item1.globalR = 4.76

        data_item2 = SearchDataItem(([-0.6, 0.7], ["e", "f"]), 0.05, None, 2)
        data_item2.globalR = 5.0

        self.characteristicsQueueGlobalR.insert(data_item1.globalR, data_item1)
        self.characteristicsQueueGlobalR.insert(data_item2.globalR, data_item2)
        getDataItem = self.characteristicsQueueGlobalR.get_best_item()[0]

        self.assertEqual(getDataItem.get_x(), 0.05)
        self.assertEqual(getDataItem.get_discrete_value_index(), 2)
        self.assertEqual(getDataItem.globalR, 5.0)
        self.assertEqual(getDataItem.get_y(), ([-0.6, 0.7], ["e", "f"]))

    def test_CanGetBestItmeWithEqualGlobalR(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.05, None, 2)
        data_item1.globalR = 5.0

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.4, None, 1)
        data_item2.globalR = 5.0

        self.characteristicsQueueGlobalR.insert(data_item1.globalR, data_item1)
        self.characteristicsQueueGlobalR.insert(data_item2.globalR, data_item2)
        get_data_item_g = self.characteristicsQueueGlobalR.get_best_item()[0]

        self.assertEqual(get_data_item_g.get_x(), 0.05)
        self.assertEqual(get_data_item_g.get_discrete_value_index(), 2)
        self.assertEqual(get_data_item_g.globalR, 5.0)
        self.assertEqual(get_data_item_g.get_y(), ([-0.6, 0.7], ['a', 'f']))

    def test_GetLen(self):
        data_item1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, 1)

        data_item2 = SearchDataItem(([-0.6, 0.7], ["e", "f"]), 0.05, 2)

        self.characteristicsQueueGlobalR.insert(data_item1.globalR, data_item1)
        self.characteristicsQueueGlobalR.insert(data_item2.globalR, data_item2)

        self.assertEqual(self.characteristicsQueueGlobalR.get_len(), 2)

    def test_GetMaxLen(self):
        self.assertEqual(self.characteristicsQueueGlobalR.get_max_len(), 3)

    def test_NotAddItemWithLowerPriority(self):
        data_item1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, None, 1)
        data_item1.globalR = 4.76

        data_item2 = SearchDataItem(([-0.6, 0.7], ["s", "k"]), 0.05, None, 2)
        data_item2.globalR = 5.0

        data_item3 = SearchDataItem(([1.08, 4.56], ["a", "b"]), 0.7, None, 1)
        data_item3.globalR = 3.05

        data_item4 = SearchDataItem(([0.076, 2.7], ["c", "d"]), 0.5, None, 1)
        data_item4.globalR = 1.5

        self.characteristicsQueueGlobalR.insert(data_item1.globalR, data_item1)
        self.characteristicsQueueGlobalR.insert(data_item2.globalR, data_item2)
        self.characteristicsQueueGlobalR.insert(data_item3.globalR, data_item3)
        self.characteristicsQueueGlobalR.insert(data_item4.globalR, data_item4)

        self.assertEqual(self.characteristicsQueueGlobalR.get_len(),
                         self.characteristicsQueueGlobalR.get_max_len())

        get_data_item1 = self.characteristicsQueueGlobalR.get_best_item()[0]
        get_data_item2 = self.characteristicsQueueGlobalR.get_best_item()[0]
        get_data_item3 = self.characteristicsQueueGlobalR.get_best_item()[0]

        self.assertEqual(self.characteristicsQueueGlobalR.get_len(), 0)

        self.assertEqual(get_data_item1.get_x(), 0.05)
        self.assertEqual(get_data_item1.globalR, 5.0)

        self.assertEqual(get_data_item2.get_x(), 0.2)
        self.assertEqual(get_data_item2.globalR, 4.76)

        self.assertEqual(get_data_item3.get_x(), 0.7)
        self.assertEqual(get_data_item3.globalR, 3.05)

    def test_AddItemWithHightPriority(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.05, 2)
        data_item1.globalR = 4.10
        self.characteristicsQueueGlobalR.insert(data_item1.globalR, data_item1)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.74, 1)
        data_item2.globalR = 2.50

        data_item3 = SearchDataItem(([1.5, 3.2], ["r", "s"]), 0.12, 0)
        data_item3.globalR = 3.23

        data_item4 = SearchDataItem(([2.67, -0.78], ["h", "k"]), 0.4, 1)
        data_item4.globalR = 3.0

        self.characteristicsQueueGlobalR.insert(data_item2.globalR, data_item2)
        self.characteristicsQueueGlobalR.insert(data_item3.globalR, data_item3)
        self.characteristicsQueueGlobalR.insert(data_item4.globalR, data_item4)

        self.assertEqual(self.characteristicsQueueGlobalR.get_len(),
                         self.characteristicsQueueGlobalR.get_max_len())
        get_data_item1 = self.characteristicsQueueGlobalR.get_best_item()[0]
        get_data_item2 = self.characteristicsQueueGlobalR.get_best_item()[0]
        get_data_item3 = self.characteristicsQueueGlobalR.get_best_item()[0]

        self.assertEqual(self.characteristicsQueueGlobalR.get_len(), 0)
        self.assertEqual(get_data_item1.get_x(), 0.05)
        self.assertEqual(get_data_item1.globalR, 4.10)

        self.assertEqual(get_data_item2.get_x(), 0.12)
        self.assertEqual(get_data_item2.globalR, 3.23)

        self.assertEqual(get_data_item3.get_x(), 0.4)
        self.assertEqual(get_data_item3.globalR, 3.0)


class TestSearchData(unittest.TestCase):
    def setUp(self):
        self.search_data = SearchData(None)

    def test_ClearQueue(self):
        try:
            self.search_data.clear_queue()
        except Exception as exc:
            assert False, f"'self.search_data.ClearQueue' raised an exception{exc}"

    def test_InsertDataItemFirst(self):
        left_data_item = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        right_data_item = SearchDataItem(([1.3, -1.78], ["e", "k"]), 1, None, 1)
        right_data_item.globalR = -3.2

        try:
            self.search_data.insert_first_data_item(left_data_item, right_data_item)
        except Exception as exc:
            assert False, f"' self.search_data.InsertDataItemFirst' raised an exception{exc}"

    def test_InsertDataItemWithRight(self):
        left_data_item = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item = SearchDataItem(([1.4, 3.078], ["c", "f"]), 0.5, None, 1)
        data_item.globalR = -0.9

        right_data_item = SearchDataItem(([-0.3, 0.78], ["a", "b"]), 1, None, 1)
        right_data_item.globalR = -3.2

        self.search_data.insert_first_data_item(left_data_item, right_data_item)

        try:
            self.search_data.insert_data_item(data_item, right_data_item)
        except Exception as exc:
            assert False, f"'self.search_data.InsertDataItem' raised an exception{exc}"

    def test_CanUseIteration(self):
        left_data_item = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item = SearchDataItem(([1.2, 2.078], ["l", "f"]), 0.5, None, 1)

        right_data_item = SearchDataItem(([0.93, -2.0], ["e", "a"]), 1, None, 1)

        self.search_data.insert_first_data_item(left_data_item, right_data_item)
        self.search_data.insert_data_item(data_item, right_data_item)

        get_data = []
        try:
            for item in self.search_data:
                get_data.append(item.get_x())
        except Exception as exc:
            assert False, f"'item.GetX' raised an exception{exc}"

        self.assertEqual(self.search_data.get_count(), 3)
        self.assertEqual(get_data[0], left_data_item.get_x())
        self.assertEqual(get_data[1], data_item.get_x())
        self.assertEqual(get_data[2], right_data_item.get_x())

    def test_FindDataItemByOneDimensionalPoint(self):
        left_data_item = SearchDataItem(([1.6, -2.78], ["e", "l"]), 0, None, 1)

        right_data_item = SearchDataItem(([-0.3, 0.78], ["a", "b"]), 1, None, 1)
        right_data_item.globalR = 1.0

        self.search_data.insert_first_data_item(left_data_item, right_data_item)

        try:
            find_right_data_item = self.search_data.find_data_item_by_one_dimensional_point(0.2)
        except Exception as exc:
            assert False, f"'self.search_data.InsertDataItem' raised an exception{exc}"

        self.assertEqual(find_right_data_item.get_x(), right_data_item.get_x())
        self.assertEqual(find_right_data_item.get_y(), right_data_item.get_y())
        self.assertEqual(find_right_data_item.globalR, right_data_item.globalR)

    def test_InsertDataItemWithoutRight(self):
        data_item1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item2 = SearchDataItem(([0.0, 1.28], ["n", "f"]), 1, None, 1)
        data_item2.globalR = 1.0

        data_item3 = SearchDataItem(([-0.4, 0.078], ["e", "f"]), 0.2, None, 1)
        data_item3.globalR = -4.76

        data_item4 = SearchDataItem(([2.09, 1.15], ["l", "l"]), 0.16, None, 1)
        data_item4.globalR = -3.6

        self.search_data.insert_first_data_item(data_item1, data_item2)
        self.search_data.insert_data_item(data_item3)
        try:
            self.search_data.insert_data_item(data_item4)
        except Exception as exc:
            assert False, f"'self.search_data.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithEqualCharactiristics(self):
        data_item_first = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item_second = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        data_item_second.globalR = 4.76

        data_item_third = SearchDataItem(([0.13, -0.2], ["e", "c"]), 0.5, None, 1)
        data_item_third.globalR = 4.76

        self.search_data.insert_first_data_item(data_item_first, data_item_second)
        try:
            self.search_data.insert_data_item(data_item_third, data_item_second)
        except Exception as exc:
            assert False, f"'self.search_data.InsertFirstDataItem'," \
                          f"'self.search_data.InsertDataItem' raised an exception{exc}"

    def test_RefillQueue(self):
        data_item_first = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item_second = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        data_item_second.globalR = 1.0

        self.search_data.insert_first_data_item(data_item_first, data_item_second)

        try:
            self.search_data.refill_queue()
        except Exception as exc:
            assert False, f"'self.search_data.RefillQueue' raised an exception{exc}"

    def test_GetDataItemWithMaxGlobalR(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.05, None, 2)
        data_item1.globalR = 4.10

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0.2, None, 1)
        data_item2.globalR = 5.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.globalR = 2.6

        self.search_data.insert_first_data_item(data_item1, data_item3)
        self.search_data.insert_data_item(data_item2)

        get_data_item = self.search_data.get_data_item_with_max_global_r()

        self.assertEqual(get_data_item.get_x(), 0.2)
        self.assertEqual(get_data_item.get_discrete_value_index(), 1)
        self.assertEqual(get_data_item.globalR, 5.0)
        self.assertEqual(get_data_item.get_y(), ([-0.3, 0.78], ['e', 'f']))

    def test_GetDataItemWithMaxGlobalRWithEaqualR(self):
        data_item1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)
        data_item2 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 1, None, 2)
        data_item2.globalR = 1.0
        data_item3 = SearchDataItem(([0.06, 0.17], ["c", "f"]), 0.078, None, 2)
        data_item3.globalR = 5.0
        data_item4 = SearchDataItem(([2.15, -4.17], ["b", "k"]), 0.7, None, 1)
        data_item4.globalR = 5.0

        self.search_data.insert_first_data_item(data_item1, data_item2)
        self.search_data.insert_data_item(data_item3, data_item2)
        self.search_data.insert_data_item(data_item4)

        get_data_item = self.search_data.get_data_item_with_max_global_r()

        self.assertEqual(get_data_item.get_x(), 0.078)
        self.assertEqual(get_data_item.get_discrete_value_index(), 2)
        self.assertEqual(get_data_item.globalR, 5.0)
        self.assertEqual(get_data_item.get_y(), ([0.06, 0.17], ['c', 'f']))

    def test_Check(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.globalR = 1.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.globalR = 2.6

        self.search_data.insert_first_data_item(data_item1, data_item2)
        self.search_data.insert_data_item(data_item3, data_item2)

        get_data_item_glob = self.search_data.get_data_item_with_max_global_r()
        get_data_item_glob2 = self.search_data.get_data_item_with_max_global_r()

        self.assertEqual(get_data_item_glob.get_x(), 0.8)
        self.assertEqual(get_data_item_glob.globalR, 2.6)

        self.assertEqual(get_data_item_glob2.get_x(), 1.0)
        self.assertEqual(get_data_item_glob2.globalR, 1.0)

    def test_GetCount(self):
        data_item_first = SearchDataItem(([-0.01, 0.78], ["e", "f"]), 0, 1)

        data_item_second = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, 1)

        self.search_data.insert_first_data_item(data_item_first, data_item_second)

        self.assertEqual(self.search_data.get_count(), 2)

    def test_GetLastItem(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.globalR = 1.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.globalR = 2.6

        self.search_data.insert_first_data_item(data_item1, data_item2)
        self.search_data.insert_data_item(data_item3, data_item2)

        get_last_item = self.search_data.get_last_item()
        self.assertEqual(get_last_item.get_x(), data_item3.get_x())

    def test_GetLastItemExcept(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.globalR = 1.0
        try:
            get_last_item = self.search_data.get_last_item()
        except Exception as exc:
            assert True, f"'self.search_data.GetLastItem()' raised an exception{exc}"


class TestSearchDataDualQueue(unittest.TestCase):
    def setUp(self):
        self.search_dataDual = SearchDataDualQueue(None, maxlen=3)

    def test_ClearQueue(self):
        try:
            self.search_dataDual.clear_queue()
        except Exception as exc:
            assert False, f"'self.search_dataDual.ClearQueue' raised an exception{exc}"

    def test_InsertDataItemWithRight(self):
        left_data_item = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item = SearchDataItem(([1.4, 3.078], ["c", "f"]), 0.5, None, 1)
        data_item.globalR = -0.9
        data_item.localR = 1.2

        right_data_item = SearchDataItem(([-0.3, 0.78], ["a", "b"]), 1, None, 1)
        right_data_item.globalR = -3.2
        right_data_item.localR = 2.03

        self.search_dataDual.insert_first_data_item(left_data_item, right_data_item)

        try:
            self.search_dataDual.insert_data_item(data_item, right_data_item)
        except Exception as exc:
            assert False, f"'self.search_dataDual.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithoutRight(self):
        data_item1 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item2 = SearchDataItem(([0.0, 1.28], ["n", "f"]), 1, None, 1)
        data_item2.globalR = 1.0
        data_item2.localR = 1.0

        data_item3 = SearchDataItem(([-0.4, 0.078], ["e", "f"]), 0.2, None, 1)
        data_item3.globalR = -4.76
        data_item3.localR = 1.34

        point = Point([2.09, 1.15], ["l", "l"])
        data_item4 = SearchDataItem(point, 0.16, None, 1)
        data_item4.globalR = 3.6
        data_item4.localR = -0.67

        self.search_dataDual.insert_first_data_item(data_item1, data_item2)
        self.search_dataDual.insert_data_item(data_item3)
        try:
            self.search_dataDual.insert_data_item(data_item4)
        except Exception as exc:
            assert False, f"'self.search_dataDual.InsertDataItem' raised an exception{exc}"

    def test_InsertDataItemWithEqualCharactiristics(self):
        data_item_first = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item_second = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        data_item_second.globalR = 4.76
        data_item_second.localR = 0.67

        point = Point([0.13, -0.2], ["e", "c"])
        data_item_third = SearchDataItem(point, 0.5, None, 1)
        data_item_third.globalR = 4.76
        data_item_third.localR = 0.67

        self.search_dataDual.insert_first_data_item(data_item_first, data_item_second)
        try:
            self.search_dataDual.insert_data_item(data_item_third, data_item_second)
        except Exception as exc:
            assert False, f"'self.search_dataDual.InsertFirstDataItem'," \
                          f"'self.search_dataDual.InsertDataItem' raised an exception{exc}"

    def test_RefillQueue(self):
        data_item_first = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 0, None, 1)

        data_item_second = SearchDataItem(([-0.5, 0.02], ["e", "c"]), 1, None, 1)
        data_item_second.globalR = 1.0
        data_item_second.localR = 1.0

        self.search_dataDual.insert_first_data_item(data_item_first, data_item_second)

        try:
            self.search_dataDual.refill_queue()
        except Exception as exc:
            assert False, f"'self.search_dataDual.RefillQueue' raised an exception{exc}"

    def test_GetDataItemWithMaxGlobalR(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.globalR = -5.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.globalR = 3.6

        data_item4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        data_item4.globalR = 2.8

        self.search_dataDual.insert_first_data_item(data_item1, data_item2)
        self.search_dataDual.insert_data_item(data_item3, data_item2)
        self.search_dataDual.insert_data_item(data_item4)

        get_data_item = self.search_dataDual.get_data_item_with_max_global_r()

        self.assertEqual(get_data_item.get_x(), 0.8)
        self.assertEqual(get_data_item.get_discrete_value_index(), 1)
        self.assertEqual(get_data_item.globalR, 3.6)
        self.assertEqual(get_data_item.get_y(), ([1.4, 3.7], ["a", "f"]))

    def test_GetDataItemWithMaxLocalR(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.localR = -5.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.localR = 3.6

        data_item4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        data_item4.localR = 2.8

        self.search_dataDual.insert_first_data_item(data_item1, data_item2)
        self.search_dataDual.insert_data_item(data_item3, data_item2)
        self.search_dataDual.insert_data_item(data_item4)

        get_data_item = self.search_dataDual.get_data_item_with_max_local_r()

        self.assertEqual(get_data_item.get_x(), 0.8)
        self.assertEqual(get_data_item.get_discrete_value_index(), 1)
        self.assertEqual(get_data_item.localR, 3.6)
        self.assertEqual(get_data_item.get_y(), ([1.4, 3.7], ["a", "f"]))

    def test_GetDataItemWithMaxGlobalRWithEqualCharacteristic(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.globalR = -5.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.globalR = 3.6

        data_item4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        data_item4.globalR = 3.6

        self.search_dataDual.insert_first_data_item(data_item1, data_item2)
        self.search_dataDual.insert_data_item(data_item3, data_item2)
        self.search_dataDual.insert_data_item(data_item4)

        get_data_item = self.search_dataDual.get_data_item_with_max_global_r()

        self.assertEqual(get_data_item.get_x(), 0.8)
        self.assertEqual(get_data_item.get_discrete_value_index(), 1)
        self.assertEqual(get_data_item.globalR, 3.6)
        self.assertEqual(get_data_item.get_y(), ([1.4, 3.7], ["a", "f"]))

    def test_GetDataItemWithMaxLocalRWithEqualCharacteristic(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.localR = -5.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.8, None, 1)
        data_item3.localR = 3.6

        data_item4 = SearchDataItem(([-3.2, 0.7], ["k", "s"]), 0.34, None, 2)
        data_item4.localR = 3.6

        self.search_dataDual.insert_first_data_item(data_item1, data_item2)
        self.search_dataDual.insert_data_item(data_item3, data_item2)
        self.search_dataDual.insert_data_item(data_item4)

        get_data_item = self.search_dataDual.get_data_item_with_max_local_r()

        self.assertEqual(get_data_item.get_x(), 0.8)
        self.assertEqual(get_data_item.get_discrete_value_index(), 1)
        self.assertEqual(get_data_item.localR, 3.6)
        self.assertEqual(get_data_item.get_y(), ([1.4, 3.7], ["a", "f"]))

    def test_CheckQueueLinked(self):
        data_item1 = SearchDataItem(([-0.6, 0.7], ["a", "f"]), 0.0, None, 2)

        data_item2 = SearchDataItem(([-0.3, 0.78], ["e", "f"]), 1.0, None, 1)
        data_item2.globalR = 1.0
        data_item2.localR = 1.0

        data_item3 = SearchDataItem(([1.4, 3.7], ["a", "f"]), 0.5, None, 1)
        data_item3.globalR = 2.6
        data_item3.localR = -1.09

        data_item4 = SearchDataItem(([-0.7, 3.1], ["s", "w"]), 0.25, None, 2)
        data_item4.globalR = 0.12
        data_item4.localR = -2.4

        data_item5 = SearchDataItem(([9.7, -0.1], ["m", "b"]), 0.76, None, 2)
        data_item5.globalR = -0.134
        data_item5.localR = 0.082

        self.search_dataDual.insert_first_data_item(data_item1, data_item2)
        self.search_dataDual.insert_data_item(data_item3, data_item2)
        get_data_item_glob = self.search_dataDual.get_data_item_with_max_global_r()

        self.assertEqual(get_data_item_glob.get_x(), 0.5)
        self.assertEqual(get_data_item_glob.globalR, 2.6)

        get_data_item_glob.globalR = -1.89
        get_data_item_glob.localR = -1.37
        self.search_dataDual.insert_data_item(data_item4, get_data_item_glob)

        get_data_item_glob2 = self.search_dataDual.get_data_item_with_max_global_r()
        self.assertEqual(get_data_item_glob2.get_x(), 1.0)
        self.assertEqual(get_data_item_glob2.globalR, 1.0)

        get_data_item_glob2.globalR = 0.5
        get_data_item_glob2.localR = 0.5
        self.search_dataDual.insert_data_item(data_item5, get_data_item_glob2)

        get_data_item_local = self.search_dataDual.get_data_item_with_max_local_r()
        self.assertEqual(get_data_item_local.get_x(), 1.0)
        self.assertEqual(get_data_item_local.localR, 0.5)


# Executing the tests in the above test case class


if __name__ == "__main__":
    unittest.main()
