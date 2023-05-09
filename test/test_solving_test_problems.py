import unittest
import numpy as np
from iOpt.problem import Problem
from problems.GKLS import GKLS
from problems.rastrigin import Rastrigin
from problems.xsquared import XSquared
from problems.hill import Hill
from problems.shekel import Shekel
from problems.grishagin import Grishagin
from problems.stronginc2 import Stronginc2
from problems.stronginc3 import Stronginc3
from problems.stronginc5 import Stronginc5
from problems.romeijn1c import Romeijn1c
from problems.romeijn2c import Romeijn2c
from problems.romeijn3c import Romeijn3c
from problems.romeijn5c import Romeijn5c
from problems.g2c import g2c
from problems.g8c import g8c
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestSolvingProblems(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""

    def setUp(self):
        self.epsVal = 0.01

    def checkIsSolved(self, problem: Problem, params: SolverParameters, number_of_global_trials: int):
        # Создаем решатель
        solver = Solver(problem, parameters=params)

        # Решение задачи
        sol = solver.Solve()

        # Проверяем что найденный АГП минимумом соответствуйте априори известному, для этой задачи, с точностью eps
        for j in range(problem.dimension):
            fabsx = np.abs(problem.knownOptimum[0].point.floatVariables[j] -
                           sol.bestTrials[0].point.floatVariables[j])
            fm = params.eps * (problem.upperBoundOfFloatVariables[j] -
                               problem.lowerBoundOfFloatVariables[j])
            self.assertLessEqual(fabsx, fm)

        # Проверяем что на решение потребовалось правильное число итераций АГП
        self.assertEqual(sol.numberOfGlobalTrials, number_of_global_trials)

    def test_Rastrigin_Solve(self):
        r = 3.5
        problem = Rastrigin(1)
        params = SolverParameters(r=r, eps=self.epsVal)
        number_of_global_trials = 44

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_XSquared_Solve(self):
        r = 3.5
        problem = XSquared(1)
        params = SolverParameters(r=r, eps=self.epsVal)
        number_of_global_trials = 36

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_Hill_Solve(self):
        r = 3.5
        problem = Hill(1)
        params = SolverParameters(r=r, eps=self.epsVal)
        number_of_global_trials = 52

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_Shekel_Solve(self):
        r = 3
        problem = Shekel(1)
        params = SolverParameters(r=r, eps=self.epsVal)

        number_of_global_trials = 34

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_Grishagin_Solve(self):
        r = 3
        problem = Grishagin(1)
        params = SolverParameters(r=r, eps=self.epsVal)

        number_of_global_trials = 543

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_GKLS_2D_Solve(self):
        # создание объекта задачи, двухмерная задача с номером 1
        problem = GKLS(2, 1)
        # Формируем параметры решателя, параметр надежности метода r=3.5, точность поиска eps=0.01
        params = SolverParameters(r=3.5, eps=0.01)

        # Необходимое число итераций алгоритма, для указанной задачи с заданными параметрами
        number_of_global_trials = 308

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_Solve_100_GKLS_2D_problem(self):
        # Необходимое число итераций алгоритма, для каждой из 100 задач, с заданными параметрами решателя
        number_of_global_trials = [883, 1441, 1061, 723, 732, 687, 684, 775, 754, 1053, 1165, 1361, 465, 760, 701, 887,
                                   1070, 1673, 1576, 744, 392, 878, 1176, 525, 1175, 856, 898, 649, 885, 771, 972, 1042,
                                   1042, 619, 796, 544, 647, 1071, 591, 833, 605, 458, 527, 1003, 473, 988, 1277, 649,
                                   531, 730, 1108, 828, 648, 221, 1502, 849, 632, 641, 609, 749, 922, 693, 991, 894,
                                   716,
                                   575, 952, 1287, 231, 1052, 625, 516, 732, 757, 617, 1455, 490, 1118, 786, 1273, 533,
                                   683,
                                   278, 1456, 1091, 1171, 974, 777, 1227, 700, 767, 728, 962, 1198, 445, 809, 946, 288,
                                   927, 903]

        for i in range(100):
            # создание объекта задачи, двухмерная задача с номером i+1
            problem = GKLS(2, i + 1)
            # Формируем параметры решателя, параметр надежности метода r=5.1, точность поиска eps=0.01
            params = SolverParameters(r=5.1, eps=0.01)

            self.checkIsSolved(problem, params, number_of_global_trials[i])

    def test_GKLS_4D_Solve(self):
        r = 3.5
        problem = GKLS(4, 3)
        params = SolverParameters(r=r, eps=self.epsVal, refineSolution=False, itersLimit=5000)

        number_of_global_trials = 2830

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_StronginC2_Solve(self):
        r = 4
        problem = Stronginc2()
        params = SolverParameters(r=r, eps=self.epsVal, epsR=0.01)

        number_of_global_trials = 437  # 373?

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_StronginC3_Solve(self):
        r = 4
        problem = Stronginc3()
        params = SolverParameters(r=r, eps=self.epsVal, epsR=0.01)

        number_of_global_trials = 512  # 555?

        self.checkIsSolved(problem, params, number_of_global_trials)

    # def test_StronginC5_Solve(self):
    #     r = ???
    #     problem = Stronginc5()
    #     params = SolverParameters(r=r, eps=self.epsVal, epsR=0.001)
    #
    #     number_of_global_trials = ???
    #
    #     self.checkIsSolved(problem, params, number_of_global_trials)

    # def test_Romeijn1c_Solve(self): # UNDEFINED OPTIMUM
    #     r = 2
    #     problem = Romeijn1c()
    #     params = SolverParameters(r=r, eps=self.epsVal, epsR=0.01)
    #
    #     number_of_global_trials = 512  # 555?
    #
    #     self.checkIsSolved(problem, params, number_of_global_trials)

    # def test_Romeijn2c_Solve(self):
    #     r = ???
    #     problem = Romeijn2c()
    #     params = SolverParameters(r=r, eps=self.epsVal, epsR=0.001)
    #
    #     number_of_global_trials = 512  # 555?
    #
    #     self.checkIsSolved(problem, params, number_of_global_trials)

    def test_Romeijn3c_Solve(self):
        r = 4
        problem = Romeijn3c()
        params = SolverParameters(r=r, eps=self.epsVal, epsR=0.01)

        number_of_global_trials = 702

        self.checkIsSolved(problem, params, number_of_global_trials)

    def test_Romeijn5c_Solve(self):
        r = 4
        problem = Romeijn5c()
        params = SolverParameters(r=r, eps=self.epsVal, epsR=0.01)

        number_of_global_trials = 193  # 189

        self.checkIsSolved(problem, params, number_of_global_trials)

    # def test_g2c_Solve(self): # UNDEFINED OPTIMUM
    #     r = ???
    #     problem = g2c()
    #     params = SolverParameters(r=r, eps=self.epsVal, epsR=0.001, itersLimit=100000)
    #
    #     number_of_global_trials =  ???
    #
    #     self.checkIsSolved(problem, params, number_of_global_trials)

    # def test_g8c_Solve(self):
    #     r = ???
    #     problem = g8c()
    #     params = SolverParameters(r=r, eps=self.epsVal, epsR=0.001, itersLimit=100000)
    #
    #     number_of_global_trials =  ???
    #
    #     self.checkIsSolved(problem, params, number_of_global_trials)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
