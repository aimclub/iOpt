from iOpt.problems.xsquared import XSquared
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticNDPaintListener

if __name__ == "__main__":
    # create the problem 2D dimension
    problem = XSquared(2)

    # add solver parameters
    params = SolverParameters(r=2.5, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticNDPaintListener("xsquared_2_2.5_0.01.png", mode='lines layers')
    solver.AddListener(apl)

    # solve the problem
    sol = solver.Solve()
