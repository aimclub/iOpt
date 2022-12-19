from iOpt.problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.listener import ConsoleFullOutputListener

if __name__ == "__main__":
    # create the problem
    problem = Rastrigin(2)

    # add solver parameters
    params = SolverParameters(r=3.5, eps=0.001)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    cfol = ConsoleFullOutputListener(mode="custom", iters=400)
    solver.AddListener(cfol)

    # solve the problem
    sol = solver.Solve()
