import math
import unittest
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestSolveRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)

    def test_solve(self):
        sol = self.solver.Solve()
        # print(sol.bestTrials)
        self.assertAlmostEqual(sol.bestTrials[0].point.floatVariables[0],
                               self.problem.knownOptimum[0].point.floatVariables[0], delta=0.05)


class TestSolveParaboloid(unittest.TestCase):
    def setUp(self):
        self.problem = XSquared(1)
        params = SolverParameters(r=3.5, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)

    def test_solve(self):
        sol = self.solver.Solve()
        # print(sol.bestTrials)
        self.assertAlmostEqual(sol.bestTrials[0].point.floatVariables[0],
                               self.problem.knownOptimum[0].point.floatVariables[0], delta=0.05)
