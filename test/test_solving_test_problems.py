import unittest
import numpy as np

from iOpt.problems.GKLS import GKLS
from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.problems.hill import Hill
from iOpt.problems.shekel import Shekel
from iOpt.problems.grishagin import Grishagin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestRastrigin(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""

    def setUp(self):
        self.epsVal = 0.01

    def test_Rastrigin_Solve(self):
        self.r = 3.5
        self.problem = Rastrigin(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 44

        sol = self.solver.Solve()

        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(self.problem.knownOptimum[0].point.floatVariables[j] -
                           sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (self.problem.upperBoundOfFloatVariables[j] -
                                self.problem.lowerBoundOfFloatVariables[j])
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_XSquared_Solve(self):
        self.r = 3.5
        self.problem = XSquared(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 36

        sol = self.solver.Solve()

        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(
                self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (
                    self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_Hill_Solve(self):
        self.r = 3.5
        self.problem = Hill(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 52

        sol = self.solver.Solve()

        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(
                self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (
                    self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_Shekel_Solve(self):
        self.r = 3
        self.problem = Shekel(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 34

        sol = self.solver.Solve()

        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(
                self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (
                    self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_Grishagin_Solve(self):
        self.r = 3
        self.problem = Grishagin(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 543

        sol = self.solver.Solve()

        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(
                self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (
                    self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_GKLS_2D_Solve(self):
        # создание объекта задачи, двухмерная задача с номером 1
        self.problem = GKLS(2, 1)
        # Формируем параметры решателя, параметр надежности метода r=3.5, точность поиска eps=0.01
        params = SolverParameters(r=3.5, eps=0.01)
        # Создаем решатель
        self.solver = Solver(self.problem, parameters=params)
        # Необходимое число итераций алгоритма, для указанной задачи с заданными параметрами
        numberOfGlobalTrials = 308

        # Решение задачи
        sol = self.solver.Solve()

        # Проверяем что найденный АГП минимумом соответствуйте априори известному, для этой задачи, с точностью eps
        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(
                self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (
                    self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
            if (fabsx > fm):
                res = res and False
        # Проверяем что решение задачи действительно сошлось к глобальному минимуму
        self.assertEqual(res, True)
        # Проверяем что на решение потребовалось правильное число итераций АГП
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_Solve_100_GKLS_2D_problem(self):
        # Необходимое число итераций алгоритма, для каждой из 100 задач, с заданными параметрами решателя
        numberOfGlobalTrials = [883, 1441, 1061, 723, 732, 687, 684, 775, 754, 1053, 1165, 1361, 465, 760, 701, 887,
                                1070, 1673, 1576, 744, 392, 878, 1176, 525, 1175, 856, 898, 649, 885, 771, 972, 1042,
                                1042, 619, 796, 544, 647, 1071, 591, 833, 605, 458, 527, 1003, 473, 988, 1277, 649, 531,
                                730, 1108, 828, 648, 221, 1502, 849, 632, 641, 609, 749, 922, 693, 991, 894, 716, 575,
                                952, 1287, 231, 1052, 625, 516, 732, 757, 617, 1455, 490, 1118, 786, 1273, 533, 683,
                                278, 1456, 1091, 1171, 974, 777, 1227, 700, 767, 728, 962, 1198, 445, 809, 946, 288,
                                927, 903]

        for i in range(100):
            # создание объекта задачи, двухмерная задача с номером i+1
            self.problem = GKLS(2, i + 1)
            # Формируем параметры решателя, параметр надежности метода r=5.1, точность поиска eps=0.01
            params = SolverParameters(r=5.1, eps=0.01)
            # Создаем решатель
            self.solver = Solver(self.problem, parameters=params)

            # Решение задачи
            sol = self.solver.Solve()

            # Проверяем что найденный АГП минимумом соответствуйте априори известному, для этой задачи, с точностью eps

            res = True
            for j in range(self.problem.dimension):
                fabsx = np.abs(
                    self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
                fm = self.epsVal * (
                        self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
                if (fabsx > fm):
                    res = res and False
            # Проверяем что решение задачи действительно сошлось к глобальному минимуму
            self.assertEqual(res, True)
            # Проверяем что на решение потребовалось правильное число итераций АГП
            self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials[i])

    def test_GKLS_4D_Solve(self):
        self.r = 3.5
        self.problem = GKLS(4, 3)
        params = SolverParameters(r=self.r, eps=self.epsVal, refineSolution=False, itersLimit=5000)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 2830

        sol = self.solver.Solve()

        res = True
        for j in range(self.problem.dimension):
            fabsx = np.abs(
                self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (
                    self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j])
            if fabsx > fm:
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()
