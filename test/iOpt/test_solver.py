import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from iOpt.problems.ga_tsp_1d import GA_TSP_1D
from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestSolveRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)

    def test_solve(self):
        sol = self.solver.Solve()
        print(sol.bestTrials)


class TestSolveParaboloid(unittest.TestCase):
    def setUp(self):
        self.problem = XSquared(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)

    def test_solve(self):
        sol = self.solver.Solve()
        print(sol.bestTrials)
