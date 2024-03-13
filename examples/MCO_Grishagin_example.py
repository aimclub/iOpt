from problems.grishagin_mco import Grishagin_mco
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
import matplotlib.pyplot as plt

if __name__ == "__main__":

    problem = Grishagin_mco(2, [3, 2])

    params = SolverParameters(r=2.5, eps=0.01, iters_limit=16000,
                              number_of_lambdas=50, start_lambdas=[[0, 1]],
                              is_scaling=False, number_of_parallel_points=2,
                              async_scheme=True)

    solver = Solver(problem=problem, parameters=params)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    sol = solver.solve()

    # output of the Pareto set (coordinates - function values)
    var = [trial.point.float_variables for trial in sol.best_trials]
    val = [[trial.function_values[i].value for i in range(2)]for trial in sol.best_trials ]
    print("size pareto set: ", len(var))
    for fvar, fval in zip(var, val):
        print(fvar, fval)

    x1 = [trial.point.float_variables[0] for trial in sol.best_trials]
    x2 = [trial.point.float_variables[1] for trial in sol.best_trials]

    plt.plot(x1, x2, 'ro')
    plt.show()

    fv1 = [trial.function_values[0].value for trial in sol.best_trials]
    fv2 = [trial.function_values[1].value for trial in sol.best_trials]

    plt.plot(fv1, fv2, 'ro')
    plt.show()
