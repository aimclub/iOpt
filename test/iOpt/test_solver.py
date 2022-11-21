import math
import unittest
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.paraboloid import Paraboloid
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import PaintListener, AnimationPaintListener

from iOpt.trial import Trial, Point


class TestSolveRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        #pl = PaintListener()
        #self.solver.AddListener(pl)
        #apl = AnimationPaintListener()
        #self.solver.AddListener(apl)

    def test_solve(self):
        sol = self.solver.Solve()
        print(sol.bestTrials[0].point.floatVariables)

class TestSolveParaboloid(unittest.TestCase):
     def setUp(self):
        self.problem = Paraboloid(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        #pl = PaintListener()
        #self.solver.AddListener(pl)
        apl = AnimationPaintListener()
        self.solver.AddListener(apl)

     def test_solve(self):
         sol = self.solver.Solve()
         print(sol.bestTrials[0].point.floatVariables)
# self.assertAlmostEqual(sol.bestTrials[0].point.floatVariables[0], self.problem.knownOptimum[0])

if __name__ == "__main__":
    unittest.main()