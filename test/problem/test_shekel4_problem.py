import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.shekel4 import Shekel4
import iOpt.problems.Shekel4.shekel4_generation as shekelGen


class TestShekel4(unittest.TestCase):
    """setUp method is overridden from the parent class Shekel4"""

    def setUp(self):
        self.fn = 1
        self.Shekel = Shekel4(self.fn)

    def test_Calculate(self):
        pointfv = np.ndarray(shape=(self.Shekel.dimension), dtype=np.double)
        pointfv.fill(0)
        point = Point(pointfv, [])
        res: np.double = 0
        for i in range(shekelGen.maxI[self.Shekel.fn - 1]):
            den: np.double = 0
            for j in range(self.Shekel.dimension):
                den = den + pow((point.floatVariables[j] - shekelGen.a[i][j]), 2.0)
            res = res - 1 / (den + shekelGen.c[i])

        functionValue = FunctionValue()
        functionValue = self.Shekel.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_OptimumValue(self):
        pointfv = np.ndarray(shape=(self.Shekel.dimension), dtype=np.double)
        pointfv.fill(4)
        point = Point(pointfv, [])
        functionValue = FunctionValue()
        functionValue = self.Shekel.Calculate(point, functionValue)
        self.assertEqual(self.Shekel.knownOptimum[0].functionValues[0].value, functionValue.value)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
