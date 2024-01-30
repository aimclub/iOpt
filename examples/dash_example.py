from problems.rastriginInt import RastriginInt
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from dashboard.static_dashboard import StaticDashboard

if __name__ == "__main__":
    # create the problem
    problem = RastriginInt(5, 2)

    # add solver parameters
    params = SolverParameters(r=2.1, eps=0.01)

    # create solver
    solver = Solver(problem, parameters=params)

    # solve problem
    solver.solve()

    # save optimization progress
    log = solver.save_progress('log_RastriginInt_5-2_2.1_0.01_1.json')

    # visualizate optimization progress
    dashboard = StaticDashboard('log_RastriginInt_5-2_2.1_0.01_1.json')
    dashboard.launch()
