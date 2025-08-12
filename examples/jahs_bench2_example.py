import numpy as np
from iOpt.output_system.listeners.static_painters import StaticDiscreteListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.jahs_bench_task import jahs_bench_task
from argparse import ArgumentParser

if __name__ == "__main__":
    """
    Minimizing the function from a set of juhs_bench_201 test tasks with visualization
    """

    # Create task object
    problem = jahs_bench_task()

    # Add argument to parser for parallel run
    parser = ArgumentParser()
    parser.add_argument('--npp', default=1, type=int)
    args = parser.parse_args()

    # Initialize parameters of solver
    method_params = SolverParameters(r=np.double(3.5), iters_limit=500,
                                     eps=np.double(0.01),
                                     number_of_parallel_points=args.npp,
                                     async_scheme=True,
                                     refine_solution=False)

    # Create solver
    solver = Solver(problem, parameters=method_params)

    # Add listener for console output
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Add visualization listener
    sdl = StaticDiscreteListener(file_name="jahs-bench.png", path_for_saves="output", mode="analysis",
                                 calc="objective function")
    solver.add_listener(sdl)

    # Solve of problem
    solver_info = solver.solve()
