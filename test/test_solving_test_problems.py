import math
import unittest
import numpy as np
import sys
from iOpt.trial import FunctionValue, FunctionValue, Point
from iOpt.problems.GKLS import GKLS
from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters


class TestRastrigin(unittest.TestCase):
    """setUp method is overridden from the parent class Rastrigin"""
    def setUp(self):        
        self.epsVal = 0.01;

    def test_Rastrigin_Solve(self):
        self.r = 3.5;
        self.problem = Rastrigin(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 44

        sol = self.solver.Solve()
                
        isSolve = 0
        res = True
        for j in range(self.problem.dimension): 
            fabsx = np.abs(self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j]);
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)

    def test_XSquared_Solve(self):
        self.r = 3.5;
        self.problem = XSquared(1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 36

        sol = self.solver.Solve()
                
        isSolve = 0
        res = True
        for j in range(self.problem.dimension): 
            fabsx = np.abs(self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j]);
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)


    def test_GKLS_2D_Solve(self):
        self.r = 3.5;
        self.problem = GKLS(2, 1)
        params = SolverParameters(r=self.r, eps=self.epsVal)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 308

        sol = self.solver.Solve()
                
        isSolve = 0
        res = True
        for j in range(self.problem.dimension): 
            fabsx = np.abs(self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j]);
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)


    def test_Solve_10_GKLS_2D_problem(self):
        self.r = 5.1;
        numberOfGlobalTrials = [883, 1441, 1061, 723, 732, 687, 684, 775, 754, 1053, 1165, 1361, 465, 760, 701, 887, 1070, 1673, 1576, 744, 392, 878, 1176, 525, 1175, 856, 898, 649, 885, 771, 972, 1042, 1042, 619, 796, 544, 647, 1071, 591, 833, 605, 458, 527, 1003, 473, 988, 1277, 649, 531, 730, 1108, 828, 648, 221, 1502, 849, 632, 641, 609, 749, 922, 693, 991, 894, 716, 575, 952, 1287, 231, 1052, 625, 516, 732, 757, 617, 1455, 490, 1118, 786, 1273, 533, 683, 278, 1456, 1091, 1171, 974, 777, 1227, 700, 767, 728, 962, 1198, 445, 809, 946, 288, 927, 903]
        
        
        for i in range(10): 
            self.problem = GKLS(2, i+1)
            params = SolverParameters(r=self.r, eps=self.epsVal)
            self.solver = Solver(self.problem, parameters=params)            

            sol = self.solver.Solve()
                
            isSolve = 0
            res = True
            for j in range(self.problem.dimension): 
                fabsx = np.abs(self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
                fm = self.epsVal * (self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j]);
                if (fabsx > fm):
                    res = res and False
            
            self.assertEqual(res, True)
            self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials[i])

    def test_GKLS_4D_Solve(self):
        self.r = 3.5;
        self.problem = GKLS(4, 3)
        params = SolverParameters(r=self.r, eps=self.epsVal, refineSolution=False, itersLimit=5000)
        self.solver = Solver(self.problem, parameters=params)
        numberOfGlobalTrials = 2830

        sol = self.solver.Solve()
                
        isSolve = 0
        res = True
        for j in range(self.problem.dimension): 
            fabsx = np.abs(self.problem.knownOptimum[0].point.floatVariables[j] - sol.bestTrials[0].point.floatVariables[j])
            fm = self.epsVal * (self.problem.upperBoundOfFloatVariables[j] - self.problem.lowerBoundOfFloatVariables[j]);
            if (fabsx > fm):
                res = res and False

        self.assertEqual(res, True)
        self.assertEqual(sol.numberOfGlobalTrials, numberOfGlobalTrials)


"""Executing the tests in the above test case class"""
if __name__ == "__main__":
    unittest.main()


