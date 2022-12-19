import unittest
from unittest import mock
from unittest.mock import Mock

import numpy as np

from iOpt.evolvent.evolvent import Evolvent

from iOpt.method.search_data import SearchData, SearchDataItem
from iOpt.solver import SolverParameters
from iOpt.method.method import Method
from iOpt.trial import Point


class TestMethod(unittest.TestCase):
    @staticmethod
    def GetImage(x: np.double) -> np.double:
        return x

    @mock.patch('iOpt.evolvent.evolvent')
    @mock.patch('iOpt.method.optim_task.OptimizationTask')
    @mock.patch('iOpt.method.search_data.SearchData')
    def setUp(self, mock_evolvent, mock_task, mock_searchData):
        mock_evolvent.GetImage.return_value = Mock(side_effect=self.GetImage)

        self.method = Method(evolvent=mock_evolvent, parameters=SolverParameters(),
                             task=mock_task, searchData=mock_searchData)

    def test_mockev(self):
        self.method.evolvent.GetImage = Mock(side_effect=self.GetImage)
        self.assertEqual(self.method.evolvent.GetImage(0.0), 0.0)
        self.assertEqual(self.method.evolvent.GetImage(0.5), 0.5)
        self.assertEqual(self.method.evolvent.GetImage(1.0), 1.0)

    def test_CalculateM_easy(self):
        left = SearchDataItem(x=0.0, y=Point(floatVariables=[5.0], discreteVariables=[]))
        curr = SearchDataItem(x=1.0, y=Point(floatVariables=[10.0], discreteVariables=[]))
        curr.SetLeft(left)
        self.method.M[0] = 1.0
        assert (self.method.M[0] == 1.0)
        curr.delta = 1.0
        left.SetZ(0.0)
        curr.SetZ(10.0)
        left.SetIndex(0)
        curr.SetIndex(0)
        # test 1. NOT AAA, BUT INITIALIZATION IS TOO BIG, AND DIFFERENT FOR easy and hard
        self.method.recalc = False
        self.method.CalculateM(curr, left)
        assert (self.method.M[0] == 10.0)
        self.assertTrue(self.method.recalc)
        # test 2
        self.method.recalc = False
        curr.SetZ(-20.0)
        self.method.CalculateM(curr, left)
        assert (self.method.M[0] == 20.0)
        self.assertTrue(self.method.recalc)
        # test 3
        self.method.recalc = False
        curr.SetZ(-5.0)
        self.method.CalculateM(curr, left)
        assert (self.method.M[0] == 20.0)
        self.assertFalse(self.method.recalc)

    def test_CalculateM_hard(self):
        left = SearchDataItem(x=0.1, y=Point(floatVariables=[6.0], discreteVariables=[]))
        curr = SearchDataItem(x=0.5, y=Point(floatVariables=[10.0], discreteVariables=[]))
        right = SearchDataItem(x=1.0, y=Point(floatVariables=[15.0], discreteVariables=[]))
        curr.SetLeft(left)
        assert (self.method.M[0] == 1.0)
        curr.delta = 0.4
        right.delta = 0.5
        left.SetZ(6.0)
        curr.SetZ(16.0)
        right.SetZ(2000.0)
        left.SetIndex(0)
        curr.SetIndex(0)
        right.SetIndex(1)
        # test 1
        self.method.recalc = False
        self.method.CalculateM(curr, left)
        self.assertEqual(self.method.M[0], 25.0)
        self.assertTrue(self.method.recalc)
        # test 2
        self.method.recalc = False
        self.method.CalculateM(right, curr)
        self.assertEqual(self.method.M[0], 25.0)
        self.assertFalse(self.method.recalc)
        # test 3
        self.method.recalc = False
        self.method.CalculateM(curr, right)
        self.assertEqual(self.method.M[0], 25.0)
        self.assertFalse(self.method.recalc)

    def test_CalculateM_dont_throws(self):
        curr = SearchDataItem(x=0.5, y=Point(floatVariables=[10.0], discreteVariables=[]))

        try:
            self.method.CalculateM(curr, None)
        except RuntimeError:
            self.fail("exception was raised!")

    def test_CalculateM_throws(self):
        curr = SearchDataItem(x=0.5, y=Point(floatVariables=[10.0], discreteVariables=[]))
        with self.assertRaises(Exception):
            self.method.CalculateM(None, curr)

    def test_Iteration_count(self):
        itcount = self.method.GetIterationsCount()
        self.method.FinalizeIteration()
        self.assertEqual(self.method.GetIterationsCount(), itcount + 1)

    def test_CalculateGlobalR(self):
        left = SearchDataItem(x=0.0, y=Point(floatVariables=[5.0], discreteVariables=[]))
        curr = SearchDataItem(x=1.0, y=Point(floatVariables=[10.0], discreteVariables=[]))
        curr.SetLeft(left)
        curr.delta = 1.0
        left.SetZ(5.0)
        curr.SetZ(15.0)
        self.method.M[0] = 10.0
        left.SetIndex(0)
        curr.SetIndex(0)
        self.method.parameters.r = 2.0
        self.method.Z[0] = 5.0

        # test 1
        self.method.CalculateGlobalR(curr, left)
        self.assertEqual(curr.globalR, 0.25)
        # test 2
        curr.SetIndex(-2)
        self.method.CalculateGlobalR(curr, left)
        self.assertEqual(curr.globalR, 2.0)
        # test 3
        curr.SetIndex(0)
        left.SetIndex(-2)
        self.method.CalculateGlobalR(curr, left)
        self.assertEqual(curr.globalR, 0)
        # test 4
        self.method.parameters.r = 4.0
        self.method.Z = [-7, -25.0]
        self.method.M = [17.54, 40.0]
        left.SetIndex(1)
        curr.SetIndex(0)
        self.method.CalculateGlobalR(curr, left)
        self.assertEqual(curr.globalR, 1.25)

    def test_CalculateGlobalR_throws(self):
        left = SearchDataItem(x=0.5, y=Point(floatVariables=[10.0], discreteVariables=[]))

        left.SetZ(15.0)
        self.method.M[0] = 10.0
        with self.assertRaises(Exception):
            self.method.CalculateGlobalR(None, left)

    def test_CalculateNextPointCoordinate(self):
        self.method.task.problem.numberOfFloatVariables = 1
        left = SearchDataItem(x=0.0, y=Point(floatVariables=[5.0], discreteVariables=[]))
        curr = SearchDataItem(x=1.0, y=Point(floatVariables=[10.0], discreteVariables=[]))

        curr.delta = 1.0
        left.SetZ(5.0)
        curr.SetZ(15.0)
        self.method.M[0] = 10.0
        left.SetIndex(0)
        curr.SetIndex(0)
        self.method.parameters.r = 2.0

        # test 1
        curr.SetLeft(left)
        self.assertEqual(0.25, self.method.CalculateNextPointCoordinate(curr))

        # test 2
        left.SetIndex(-2)
        self.assertEqual(0.5, self.method.CalculateNextPointCoordinate(curr))

    def test_CalculateNextPointCoordinate_throws(self):
        self.method.task.problem.numberOfFloatVariables = 1
        curr = SearchDataItem(x=0.5, y=Point(floatVariables=[10.0], discreteVariables=[]))

        curr.SetZ(15.0)
        self.method.M[0] = 10.0
        # test 1
        with self.assertRaises(Exception):
            self.method.CalculateNextPointCoordinate(curr)
        # test 2
        curr.SetLeft(curr)
        with self.assertRaises(Exception):
            self.method.CalculateNextPointCoordinate(curr)


# def test_RecalcAll_mock(self):


# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()
