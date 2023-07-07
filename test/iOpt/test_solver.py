import math
import unittest
import numpy as np

from problems.rastrigin import Rastrigin
from problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestSolveRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)

    def test_solve(self):
        sol = self.solver.solve()
        # print(sol.best_trials)
        self.assertAlmostEqual(sol.best_trials[0].point.float_variables[0],
                               self.problem.known_optimum[0].point.float_variables[0], delta=0.05)


class TestSolveParaboloid(unittest.TestCase):
    def setUp(self):
        self.problem = XSquared(1)
        params = SolverParameters(r=3.5, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)

    def test_solve(self):
        sol = self.solver.solve()
        # print(sol.best_trials)
        self.assertAlmostEqual(sol.best_trials[0].point.float_variables[0],
                               self.problem.known_optimum[0].point.float_variables[0], delta=0.05)
