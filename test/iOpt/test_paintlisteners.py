import unittest

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import PaintListener, AnimationPaintListener


class TestStaticPaintRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        pl = PaintListener()
        self.solver.AddListener(pl)

    def test_solveWithPrint(self):
        sol = self.solver.Solve()

class TestStaticPaintXSquared(unittest.TestCase):
     def setUp(self):
        self.problem = XSquared(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        apl = AnimationPaintListener()
        self.solver.AddListener(apl)

     def test_solveWithPrint(self):
         sol = self.solver.Solve()

if __name__ == "__main__":
    unittest.main()

