from problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.animate_painters import AnimatePainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    # create the problem 3D dimension
    problem = XSquared(2)

    # add solver parameters
    params = SolverParameters(r=2.1, eps=0.01, numberOfParallelPoints=8)

    # create solver
    solver = Solver(problem, parameters=params)

    cfol = ConsoleOutputListener(mode="full")
    solver.AddListener(cfol)

    # add needed listeners for solver
    apl = AnimatePainterNDListener("xsquared_1_2.5_0.01.png")
    solver.AddListener(apl)

    # solve the problem
    sol = solver.Solve()