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
        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=50, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol50 = self.solver.solve()
        self.solver.save_progress('Rastrigin2_50.json', mode='full')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.load_progress('Rastrigin2_50.json', mode='full')
        self.sol50_100 = self.solver.solve()
        self.solver.save_progress('Rastrigin2_50_100.json', mode='full')


        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.solve()
        self.solver.save_progress('Rastrigin2_100.json', mode='full')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.solve()

        with open('Rastrigin2_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('Rastrigin2_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.best_trials, self.sol100_ns.best_trials)
        self.assertEqual(data1['best_trials'], data2['best_trials'])
        self.assertEqual(data1['SearchDataItem'], data2['SearchDataItem'])
        self.assertEqual(data1['solution'][0]['number_of_global_trials'],
                         data2['solution'][0]['number_of_global_trials'])
        self.assertEqual(data1['solution'][0]['number_of_local_trials'],
                         data2['solution'][0]['number_of_local_trials'])
        self.assertEqual(data1['solution'][0]['solution_accuracy'],
                         data2['solution'][0]['solution_accuracy'])

        pathlist = ['Rastrigin2_50.json', 'Rastrigin2_50_100.json', 'Rastrigin2_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)

    def test_RastriginInt(self):
        self.problem = RastriginInt(5, 2)
        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=50, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol50 = self.solver.solve()
        self.solver.save_progress('RastriginInt_50.json', mode='only search_data')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.load_progress('RastriginInt_50.json', mode='only search_data')
        self.sol50_100 = self.solver.solve()
        self.solver.save_progress('RastriginInt_50_100.json', mode='only search_data')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.solve()
        self.solver.save_progress('RastriginInt_100.json', mode='only search_data')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.solve()

        with open('RastriginInt_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('RastriginInt_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.best_trials, self.sol100_ns.best_trials)
        self.assertEqual(data1['best_trials'], data2['best_trials'])
        self.assertEqual(data1['SearchDataItem'], data2['SearchDataItem'])

        pathlist = ['RastriginInt_50.json', 'RastriginInt_50_100.json', 'RastriginInt_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)


    def test_GKLS(self):
        self.problem = GKLS(3, 2)
        self.params = SolverParameters(r=3.5, eps=0.01, iters_limit=50, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol50 = self.solver.solve()
        self.solver.save_progress('GKLS_50.json')

        self.params = SolverParameters(r=3.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.load_progress('GKLS_50.json')
        self.sol50_100 = self.solver.solve()
        self.solver.save_progress('GKLS_50_100.json')


        self.params = SolverParameters(r=3.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.solve()
        self.solver.save_progress('GKLS_100.json')

        self.params = SolverParameters(r=3.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.solve()

        with open('GKLS_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('GKLS_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.best_trials, self.sol100_ns.best_trials)
        self.assertEqual(data1['best_trials'], data2['best_trials'])
        self.assertEqual(data1['SearchDataItem'], data2['SearchDataItem'])

        pathlist = ['GKLS_50.json', 'GKLS_50_100.json', 'GKLS_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)

    def test_Stronginc2(self):
        self.problem = Stronginc2()
        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=50, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol50 = self.solver.solve()
        self.solver.save_progress('Stronginc2_50.json')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.solver.load_progress('Stronginc2_50.json')
        self.sol50_100 = self.solver.solve()
        self.solver.save_progress('Stronginc2_50_100.json')


        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100 = self.solver.solve()
        self.solver.save_progress('Stronginc2_100.json')

        self.params = SolverParameters(r=2.5, eps=0.01, iters_limit=100, refine_solution=False)
        self.solver = Solver(self.problem, parameters=self.params)
        self.sol100_ns = self.solver.solve()

        with open('Stronginc2_50_100.json') as json_file:
            data1 = json.load(json_file)

        with open('Stronginc2_100.json') as json_file:
            data2 = json.load(json_file)

        self.assertEqual(self.sol50_100.best_trials, self.sol100_ns.best_trials)
        self.assertEqual(data1['best_trials'], data2['best_trials'])
        self.assertEqual(data1['SearchDataItem'], data2['SearchDataItem'])

        pathlist = ['Stronginc2_50.json', 'Stronginc2_50_100.json', 'Stronginc2_100.json']
        for path in pathlist:
            if os.path.isfile(path):
                os.remove(path)






# Executing the tests in the above test case class


if __name__ == "__main__":
    unittest.main()
