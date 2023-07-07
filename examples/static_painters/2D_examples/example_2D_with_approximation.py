from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterNDListener

if __name__ == "__main__":
    # create the problem 2D dimension
    problem = Rastrigin(2)

    # add solver parameters
    params = SolverParameters(r=2.5, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticPainterNDListener("rastrigin_2_2.5_0.01_approx.png", mode='surface', calc='approximation')
    solver.add_listener(apl)

    # solve the problem
    sol = solver.solve()
