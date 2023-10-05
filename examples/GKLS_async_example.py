from problems.GKLS import GKLS
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener


def solve_all_gkls():
    for i in range(1, 2):
        problem = GKLS(dimension=3, functionNumber=i)
        params = SolverParameters(r=4, eps=0.01, number_of_parallel_points=4, async_scheme=True)
        solver = Solver(problem=problem, parameters=params)
        cfol = ConsoleOutputListener(mode='result')
        solver.add_listener(cfol)
        solver.solve()


if __name__ == "__main__":
    solve_all_gkls()
