import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point

from problems.jahs_bench_task import jahs_bench_task


class TestJahsBenchProblem(unittest.TestCase):

    def test_JahsBench(self):
        problem = jahs_bench_task()

        first_point = Point(np.array([0.50098779, 0.00500988]), np.array(['ReLU']))
        last_point = Point(np.array([0.45513525, 0.00120509]), np.array(['Mish']))

        first_value = FunctionValue()
        last_value = FunctionValue()
        first_value = problem.calculate(first_point, first_value)
        last_value = problem.calculate(last_point, last_value)

        FIRST_VALUE_CHECK = 83.86813163757324
        LAST_VALUE_CHECK = 15.652191162109375

        self.assertEqual(first_value.value, FIRST_VALUE_CHECK)
        self.assertEqual(last_value.value, LAST_VALUE_CHECK)

        print(problem.name, "is OK")


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
