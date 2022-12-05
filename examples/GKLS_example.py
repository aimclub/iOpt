import math
import unittest
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.problems.GKLS import GKLS
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

problem = GKLS(4)
#problem=Rastrigin(2)
#problem=XSquared(2)
params = SolverParameters(r=3.5, eps=0.01, itersLimit=10000, refineSolution=True)
solver = Solver(problem, parameters=params)

sol = solver.Solve()
print(sol.numberOfGlobalTrials)
print(sol.numberOfLocalTrials)
print(sol.solvingTime)

print(problem.knownOptimum[0].point.floatVariables)
print(sol.bestTrials[0].point.floatVariables)
print(sol.bestTrials[0].functionValues[0].value)