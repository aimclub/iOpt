import math
import unittest
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.paraboloid import Paraboloid
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import ConsoleFullOutputListener


class TestConsoleOutput(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        cfol = ConsoleFullOutputListener()
        self.solver.AddListener(cfol)

    def test_solveWithConsalOutput(self):
        sol = self.solver.Solve()
        #print(sol.bestTrials[0].point.floatVariables)

if __name__ == "__main__":
    unittest.main()