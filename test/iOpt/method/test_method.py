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
                             task=mock_task, search_data=mock_searchData)

    def test_mockev(self):
        self.method.evolvent.GetImage = Mock(side_effect=self.GetImage)
        self.assertEqual(self.method.evolvent.GetImage(0.0), 0.0)
        self.assertEqual(self.method.evolvent.GetImage(0.5), 0.5)
        self.assertEqual(self.method.evolvent.GetImage(1.0), 1.0)

    def test_CalculateM_easy(self):
        left = SearchDataItem(x=0.0, y=Point(float_variables=[5.0], discrete_variables=[]))
        curr = SearchDataItem(x=1.0, y=Point(float_variables=[10.0], discrete_variables=[]))
        curr.set_left(left)
        self.method.M[0] = 1.0
        assert (self.method.M[0] == 1.0)
        curr.delta = 1.0
        left.set_z(0.0)
        curr.set_z(10.0)
        left.set_index(0)
        curr.set_index(0)
        # test 1. NOT AAA, BUT INITIALIZATION IS TOO BIG, AND DIFFERENT FOR easy and hard
        self.method.recalcR = False
        self.method.calculate_m(curr, left)
        assert (self.method.M[0] == 10.0)
        self.assertTrue(self.method.recalcR)
        # test 2
        self.method.recalcR = False
        curr.set_z(-20.0)
        self.method.calculate_m(curr, left)
        assert (self.method.M[0] == 20.0)
        self.assertTrue(self.method.recalcR)
        # test 3
        self.method.recalcR = False
        curr.set_z(-5.0)
        self.method.calculate_m(curr, left)
        assert (self.method.M[0] == 20.0)
        self.assertFalse(self.method.recalcR)

    def test_CalculateM_hard(self):
        left = SearchDataItem(x=0.1, y=Point(float_variables=[6.0], discrete_variables=[]))
        curr = SearchDataItem(x=0.5, y=Point(float_variables=[10.0], discrete_variables=[]))
        right = SearchDataItem(x=1.0, y=Point(float_variables=[15.0], discrete_variables=[]))
        curr.set_left(left)
        assert (self.method.M[0] == 1.0)
        curr.delta = 0.4
        right.delta = 0.5
        left.set_z(6.0)
        curr.set_z(16.0)
        right.set_z(2000.0)
        left.set_index(0)
        curr.set_index(0)
        right.set_index(1)
        # test 1
        self.method.recalcR = False
        self.method.calculate_m(curr, left)
        self.assertEqual(self.method.M[0], 25.0)
        self.assertTrue(self.method.recalcR)
        # test 2
        self.method.recalcR = False
        self.method.calculate_m(right, curr)
        self.assertEqual(self.method.M[0], 25.0)
        self.assertFalse(self.method.recalcR)
        # test 3
        self.method.recalcR = False
        self.method.calculate_m(curr, right)
        self.assertEqual(self.method.M[0], 25.0)
        self.assertFalse(self.method.recalcR)

    def test_CalculateM_dont_throws(self):
        curr = SearchDataItem(x=0.5, y=Point(float_variables=[10.0], discrete_variables=[]))

        try:
            self.method.calculate_m(curr, None)
        except RuntimeError:
            self.fail("exception was raised!")

    def test_CalculateM_throws(self):
        curr = SearchDataItem(x=0.5, y=Point(float_variables=[10.0], discrete_variables=[]))
        with self.assertRaises(Exception):
            self.method.calculate_m(None, curr)

    def test_Iteration_count(self):
        itcount = self.method.get_iterations_count()
        self.method.finalize_iteration()
        self.assertEqual(self.method.get_iterations_count(), itcount + 1)

    def test_calculate_global_r(self):
        left = SearchDataItem(x=0.0, y=Point(float_variables=[5.0], discrete_variables=[]))
        curr = SearchDataItem(x=1.0, y=Point(float_variables=[10.0], discrete_variables=[]))
        curr.set_left(left)
        curr.delta = 1.0
        left.set_z(5.0)
        curr.set_z(15.0)
        self.method.M[0] = 10.0
        left.set_index(0)
        curr.set_index(0)
        self.method.parameters.r = 2.0
        self.method.Z[0] = 5.0

        # test 1
        self.method.calculate_global_r(curr, left)
        self.assertEqual(curr.globalR, 0.25)
        # test 2
        curr.set_index(-2)
        self.method.calculate_global_r(curr, left)
        self.assertEqual(curr.globalR, 2.0)
        # test 3
        curr.set_index(0)
        left.set_index(-2)
        self.method.calculate_global_r(curr, left)
        self.assertEqual(curr.globalR, 0)
        # test 4
        self.method.parameters.r = 4.0
        self.method.Z = [-7, -25.0]
        self.method.M = [17.54, 40.0]
        left.set_index(1)
        curr.set_index(0)
        self.method.calculate_global_r(curr, left)
        self.assertEqual(curr.globalR, 1.25)

    def test_calculate_global_r_throws(self):
        left = SearchDataItem(x=0.5, y=Point(float_variables=[10.0], discrete_variables=[]))

        left.set_z(15.0)
        self.method.M[0] = 10.0
        with self.assertRaises(Exception):
            self.method.calculate_global_r(None, left)

    def test_CalculateNextPointCoordinate(self):
        self.method.task.problem.number_of_float_variables = 1
        left = SearchDataItem(x=0.0, y=Point(float_variables=[5.0], discrete_variables=[]))
        curr = SearchDataItem(x=1.0, y=Point(float_variables=[10.0], discrete_variables=[]))

        curr.delta = 1.0
        left.set_z(5.0)
        curr.set_z(15.0)
        self.method.M[0] = 10.0
        left.set_index(0)
        curr.set_index(0)
        self.method.parameters.r = 2.0

        # test 1
        curr.set_left(left)
        self.assertEqual(0.25, self.method.calculate_next_point_coordinate(curr))

        # test 2
        left.set_index(-2)
        self.assertEqual(0.5, self.method.calculate_next_point_coordinate(curr))

    def test_CalculateNextPointCoordinate_throws(self):
        self.method.task.problem.number_of_float_variables = 1
        curr = SearchDataItem(x=0.5, y=Point(float_variables=[10.0], discrete_variables=[]))

        curr.set_z(15.0)
        self.method.M[0] = 10.0
        # test 1
        with self.assertRaises(Exception):
            self.method.calculate_next_point_coordinate(curr)
        # test 2
        curr.set_left(curr)
        with self.assertRaises(Exception):
            self.method.calculate_next_point_coordinate(curr)


# def test_RecalcAll_mock(self):


# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()
