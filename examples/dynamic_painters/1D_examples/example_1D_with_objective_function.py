from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.animate_painters import AnimatePainterListener

if __name__ == "__main__":
    # create the problem 1D dimension
    problem = Rastrigin(1)

    # add solver parameters
    params = SolverParameters(r=2.5, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    apl = AnimatePainterListener("rastrigin_1_2.5_0.01.png")
    solver.add_listener(apl)

    # solve the problem
    sol = solver.solve()
