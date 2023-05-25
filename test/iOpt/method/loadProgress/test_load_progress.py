import json
import unittest

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.GKLS import GKLS
from problems.rastrigin import Rastrigin
from problems.stronginc2 import Stronginc2


class TestLoadProgress(unittest.TestCase):
    def setUp(self):
        self.problem = None
        self.params = SolverParameters()
        self.solver = None

    def test_Rastrigin2(self):
        self.problem = Rastrigin(2)
        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=50, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('Rastrigin2_50.json')

        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.LoadProgress('Rastrigin2_50.json')
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('Rastrigin2_50_100.json')


        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('Rastrigin2_100.json')

        with open('Rastrigin2_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('Rastrigin2_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(data1, data2)

    def test_GKLS(self):
        self.problem = GKLS(3, 2)
        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=50, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('GKLS_50.json')

        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.LoadProgress('GKLS_50.json')
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('GKLS_50_100.json')


        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('GKLS_100.json')

        with open('GKLS_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('GKLS_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(data1, data2)

    def test_Stronginc2(self):
        self.problem = Stronginc2()
        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=50, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('Stronginc2_50.json')

        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.LoadProgress('Stronginc2_50.json')
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('Stronginc2_50_100.json')


        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol = self.solver.Solve()
        self.solver.SaveProgress('Stronginc2_100.json')

        with open('Stronginc2_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('Stronginc2_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(data1, data2)



# Executing the tests in the above test case class


if __name__ == "__main__":
    unittest.main()
