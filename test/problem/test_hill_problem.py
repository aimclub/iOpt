import math
import unittest
import numpy as np
from iOpt.trial import FunctionValue
from iOpt.trial import Point
from iOpt.problems.hill import Hill


class TestHill(unittest.TestCase):
    """setUp method is overridden from the parent class TestHill"""

    def setUp(self):
        self.Hill = Hill(0)

    def test_Calculate(self):
        point = Point([0.0], [])
        res: np.double = 0
        aHill = np.array(
            [0.980773, -0.794122, -0.203711, 0.394696, -0.0102847, 0.53679, -0.609485, 0.870846, 0.726005, 0.00979644,
             -0.601245, -0.853633, 0.25248, 0.608936],
            dtype=np.double)
        bHill = np.array(
            [-0.876339, -0.629688, 0.695303, 0.00503555, 0.783258, -0.358318, 0.541063, -0.742302, 0.346843, -0.088168,
             0.0708335, -0.0198675, -0.75164, 0.989074],
            dtype=np.double)
        NUM_HILL_COEFF = 14
        for i in range(NUM_HILL_COEFF):
            res = res + aHill[i] * math.sin(2 * i * math.pi * point.floatVariables[0]) + bHill[i] * math.cos(
                2 * i * math.pi * point.floatVariables[0])

        functionValue = FunctionValue()
        functionValue = self.Hill.Calculate(point, functionValue)
        self.assertEqual(functionValue.value, res)

    def test_OptimumValue(self):
        self.assertEqual(self.Hill.knownOptimum[0].functionValues[0].value, -4.8322090)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
