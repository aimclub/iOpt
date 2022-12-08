import math
import unittest
import sys
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener, AnimationPaintListener, StaticNDPaintListener, AnimationNDPaintListener, ConsoleFullOutputListener

from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":
    """
    Запуск решения с визуализацией задачи Растригина с визуализацией
    """

    problem = Rastrigin(1)
    params = SolverParameters(r=3.5, eps=0.01, itersLimit=100, refineSolution=True)
    solver = Solver(problem, parameters=params)

    pl = StaticPaintListener("rastrigin.png", "output", isPointsAtBottom = False)
    apl = AnimationPaintListener("rastriginAnim.png", "output", isPointsAtBottom = False, toPaintObjFunc=True)
    solver.AddListener(pl)
    solver.AddListener(apl)

    sol = solver.Solve()
    print(sol.numberOfGlobalTrials)
    print(sol.numberOfLocalTrials)
    print(sol.solvingTime)

    print(problem.knownOptimum[0].point.floatVariables)
    print(sol.bestTrials[0].point.floatVariables)
    print(sol.bestTrials[0].functionValues[0].value)
