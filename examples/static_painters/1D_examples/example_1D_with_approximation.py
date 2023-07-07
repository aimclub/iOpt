from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterListener

if __name__ == "__main__":
    # create the problem 1D dimension
    problem = Rastrigin(1)

    # add solver parameters
    params = SolverParameters(r=3, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = StaticPainterListener("rastrigin_1_3_0.01.png", mode="approximation")
    solver.add_listener(apl)

    # solve the problem
    sol = solver.solve()
    