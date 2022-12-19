from iOpt.problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import StaticPaintListener
from iOpt.method.listener import ConsoleFullOutputListener

if __name__ == "__main__":
    # create the problem 2D dimension
    problem = Rastrigin(1)

    # add solver parameters
    params = SolverParameters(r=2.5, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticPaintListener("rastrigin_1_2.5_0.01.png", mode="interpolation")
    solver.AddListener(apl)

    # solve the problem
    sol = solver.Solve()
