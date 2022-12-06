import unittest

from iOpt.method.listener import ConsoleFullOutputListener
from iOpt.problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestStaticConsoleOutput(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        cfol = ConsoleFullOutputListener(mode=1)
        self.solver.AddListener(cfol)

    def test_solveWithConsalOutput(self):
        sol = self.solver.Solve()


class TestDynamicConsoleOutput(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        cfol = ConsoleFullOutputListener(mode=2)
        self.solver.AddListener(cfol)

    def test_solveWithConsalOutput(self):
        sol = self.solver.Solve()


if __name__ == "__main__":
    unittest.main()
