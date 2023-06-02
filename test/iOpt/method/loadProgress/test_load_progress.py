import json
import unittest
import os

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.GKLS import GKLS
from problems.rastrigin import Rastrigin
from problems.rastriginInt import RastriginInt
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
        self.sol50 = self.solver.Solve()
        self.solver.SaveProgress('Rastrigin2_50.json')

        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.LoadProgress('Rastrigin2_50.json')
        self.sol50_100 = self.solver.Solve()
        self.solver.SaveProgress('Rastrigin2_50_100.json')


        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.Solve()
        self.solver.SaveProgress('Rastrigin2_100.json')

        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.Solve()

        with open('Rastrigin2_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('Rastrigin2_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.bestTrials, self.sol100_ns.bestTrials)
        self.assertEqual(data1, data2)

        pathlist = ['Rastrigin2_50.json', 'Rastrigin2_50_100.json', 'Rastrigin2_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)


    def test_GKLS(self):
        self.problem = GKLS(3, 2)
        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=50, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol50 = self.solver.Solve()
        self.solver.SaveProgress('GKLS_50.json')

        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.LoadProgress('GKLS_50.json')
        self.sol50_100 = self.solver.Solve()
        self.solver.SaveProgress('GKLS_50_100.json')


        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.Solve()
        self.solver.SaveProgress('GKLS_100.json')

        self.params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.Solve()

        with open('GKLS_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('GKLS_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.bestTrials, self.sol100_ns.bestTrials)
        self.assertEqual(data1, data2)

        pathlist = ['GKLS_50.json', 'GKLS_50_100.json', 'GKLS_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)

    def test_Stronginc2(self):
        self.problem = Stronginc2()
        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=50, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol50 = self.solver.Solve()
        self.solver.SaveProgress('Stronginc2_50.json')

        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.LoadProgress('Stronginc2_50.json')
        self.sol50_100 = self.solver.Solve()
        self.solver.SaveProgress('Stronginc2_50_100.json')


        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.Solve()
        self.solver.SaveProgress('Stronginc2_100.json')

        self.params = SolverParameters(r=2.5, eps=0.01, itersLimit=100, refineSolution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.Solve()

        with open('Stronginc2_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('Stronginc2_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.bestTrials, self.sol100_ns.bestTrials)
        self.assertEqual(data1, data2)

        pathlist = ['Stronginc2_50.json', 'Stronginc2_50_100.json', 'Stronginc2_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)






# Executing the tests in the above test case class


if __name__ == "__main__":
    unittest.main()
