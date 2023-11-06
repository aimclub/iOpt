from problems.rastrigin import Rastrigin
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener

if __name__ == "__main__":
    # create the problem
    problem = Rastrigin(1)

    # add solver parameters
    params = SolverParameters(r=3.5, eps=0.001)

    # create solver
    solver = Solver(problem, parameters=params)

    # add needed listeners for solver
    cfol = ConsoleOutputListener(mode="full")
    solver.add_listener(cfol)

    # solve the problem
    sol = solver.solve()

