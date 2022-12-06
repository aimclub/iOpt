import unittest

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener, AnimationPaintListener, StaticNDPaintListener, AnimationNDPaintListener
from iOpt.method.listener import ConsoleFullOutputListener

class TestStaticPaintRastrigin(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        pl = StaticPaintListener("output", "rastrigin_1_3.5_0.001.png")
        self.solver.AddListener(pl)

    def test_solveWithPrint(self):
        sol = self.solver.Solve()

class TestStaticPaintRastrigin2D(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(2)
        params = SolverParameters(r=3, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)
        apl = StaticNDPaintListener("output", "rastrigin_2_3.5_0.001.png", varsIndxs = [0,1])
        self.solver.AddListener(apl)

    def test_solve(self):
        sol = self.solver.Solve()

class TestOneParamPaintRastrigin2D(unittest.TestCase):
    def setUp(self):
        self.problem = Rastrigin(2)
        params = SolverParameters(r=3.5, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)
        apl_0 = StaticPaintListener("output", "rastrigin_2_3.5_0.001_static1D_0.png", indx = 0)
        apl_1 = StaticPaintListener("output", "rastrigin_2_3.5_0.001_static1D_1.png", indx = 1, isPointsAtBottom=False)
        self.solver.AddListener(apl_0)
        self.solver.AddListener(apl_1)

    def test_solve(self):
        sol = self.solver.Solve()

class TestAnimatePaintXSquared(unittest.TestCase):
     def setUp(self):
        self.problem = XSquared(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        apl = AnimationPaintListener("output", "xsquared_1_3.5_0.001.png")
        self.solver.AddListener(apl)

     def test_solveWithPrint(self):
         sol = self.solver.Solve()

class TestAnimatePaintRastrigin(unittest.TestCase):
     def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        apl = AnimationPaintListener("output", "rastrigin_1_3.5_0.001_anim.png", isPointsAtBottom=False)
        self.solver.AddListener(apl)

     def test_solveWithPrint(self):
         sol = self.solver.Solve()

class TestAnimatePaintXSquared2D(unittest.TestCase):
    def setUp(self):
        self.problem = XSquared(2)
        params = SolverParameters(r=3.0, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)

        apl = AnimationNDPaintListener("output", "xsquared_2_3.0_0.01.png", varsIndxs=[0,1])
        self.solver.AddListener(apl)

    def test_solve(self):
        sol = self.solver.Solve()



class TestAnimatePaintRastriginWithoutOF(unittest.TestCase):
     def setUp(self):
        self.problem = Rastrigin(1)
        params = SolverParameters(r=3.5, eps=0.001)
        self.solver = Solver(self.problem, parameters=params)
        apl = AnimationPaintListener("output", "rastrigin_1_3.5_0.001_anim_wof.png", isPointsAtBottom=False, toPaintObjFunc=False)
        self.solver.AddListener(apl)
        spl = StaticPaintListener("", "rastrigin_1_3.5_0.001_stat_wof.png", isPointsAtBottom=False, toPaintObjFunc=False)
        self.solver.AddListener(spl)

     def test_solveWithPrint(self):
         sol = self.solver.Solve()

class TestAnimatePaintXSquared2DWithoutOF(unittest.TestCase):
    def setUp(self):
        self.problem = XSquared(2)
        params = SolverParameters(r=3.0, eps=0.01)
        self.solver = Solver(self.problem, parameters=params)

        apl = AnimationNDPaintListener("output", "xsquared_2_3.0_0.01_anim_wof.png", varsIndxs=[0,1], toPaintObjFunc=False)
        self.solver.AddListener(apl)
        spl = StaticNDPaintListener("C:\\GitHub\\Actual\\iOpt\\output", "xsquared_2_3.0_0.01_stats_wof.png", varsIndxs=[0,1], toPaintObjFunc=False)
        self.solver.AddListener(spl)

    def test_solve(self):
        sol = self.solver.Solve()

if __name__ == "__main__":
    unittest.main()

