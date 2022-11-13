import math
import unittest
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.paraboloid import Paraboloid
from iOpt.solver import Solver


# class TestSolveRastrigin(unittest.TestCase):
#     def setUp(self):
#         self.problem = Rastrigin(1)
#         self.solver = Solver(self.problem)
#
#     def test_solve(self):
#         sol = self.solver.Solve()
#         print(sol.bestTrials)


class TestSolveParaboloid(unittest.TestCase):
    def setUp(self):
        self.problem = Paraboloid(1)
        self.solver = Solver(self.problem)

    def test_solve(self):
        sol = self.solver.Solve()
        print(sol.bestTrials)
        # self.assertAlmostEqual(sol.bestTrials[0].point.floatVariables[0], self.problem.knownOptimum[0])
