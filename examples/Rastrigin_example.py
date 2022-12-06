import math
import unittest
import sys
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener, AnimationPaintListener, StaticNDPaintListener, AnimationNDPaintListener
from iOpt.method.listener import ConsoleFullOutputListener

#import subprocess
from subprocess import Popen, PIPE, STDOUT

problem = Rastrigin(1)
params = SolverParameters(r=3.5, eps=0.01, itersLimit=10, refineSolution=True)
#params = SolverParameters(r=3.5, eps=0.01, refineSolution=True)
#params = SolverParameters(r=3.5, eps=0.01, itersLimit=10)
solver = Solver(problem, parameters=params)

pl = StaticPaintListener("output", "rastrigin.png", isPointsAtBottom = False, toPaintObjFunc=True)
apl = AnimationPaintListener("output", "rastriginAnim.png", isPointsAtBottom = False, toPaintObjFunc=True)
solver.AddListener(pl)
solver.AddListener(apl)

sol = solver.Solve()
print(sol.numberOfGlobalTrials)
print(sol.numberOfLocalTrials)
print(sol.solvingTime)

print(problem.knownOptimum[0].point.floatVariables)
print(sol.bestTrials[0].point.floatVariables)
print(sol.bestTrials[0].functionValues[0].value)

aa = 0