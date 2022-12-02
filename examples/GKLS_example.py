import math
import unittest
import numpy as np

from iOpt.problems.rastrigin import Rastrigin
from iOpt.problems.xsquared import XSquared
from iOpt.problems.GKLS import GKLS
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

problem = GKLS(2)
#problem=Rastrigin(2)
params = SolverParameters(r=3.5, eps=0.001, itersLimit=10)
solver = Solver(problem, parameters=params)

sol = solver.Solve()
print(sol.numberOfGlobalTrials)
print(sol.solvingTime)

print(sol.bestTrials[0].point.floatVariables)
print(sol.bestTrials[0].functionValues[0].value)