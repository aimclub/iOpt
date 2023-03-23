from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener

if __name__ == "__main__":
    # create the problem 1D dimension
    problem = Rastrigin(1)

    # add solver parameters
    params = SolverParameters(r=3, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticPaintListener("rastrigin_1_3_0.01.png", mode="approximation")
    solver.AddListener(apl)

    # solve the problem
    sol = solver.Solve()
    