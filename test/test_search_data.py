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

# Executing the tests in the above test case class
if __name__ == "__main__":
 unittest.main()