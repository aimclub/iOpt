import numpy as np
from iOpt.output_system.listeners.static_painters import StaticPainterListener
from iOpt.output_system.listeners.animate_painters import AnimatePainterListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from problems.jahs_bench_task import jahs_bench_task
from argparse import ArgumentParser

if __name__ == "__main__":
    """
    Minimizing the function from a set of juhs_bench_201 test tasks
    """

    # Create task object
    problem = jahs_bench_task()

    # Add argument to parser for parallel run
    parser = ArgumentParser()
    parser.add_argument('--npp', default=1, type=int)
    args = parser.parse_args()

    # Initialize parameters of solver
    method_params = SolverParameters(r=np.double(3), iters_limit=1500,
                                     eps=np.double(0.01),
                                     number_of_parallel_points=args.npp,
                                     async_scheme=True,
                                     refine_solution=False)

    # Create solver
    solver = Solver(problem, parameters=method_params)

    # Add listener for console output
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Solve of problem
    solver_info = solver.solve()
