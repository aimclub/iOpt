from problems.mco_test1 import mco_test1
from problems.mco_test3 import mco_test3
from problems.mco_test5 import mco_test5
from problems.mco_test6 import mco_test6
from problems.grishagin_mco import Grishagin_mco
from iOpt.trial import Point
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.static_painters import StaticPainterListener
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.output_system.listeners.static_painters import StaticPainterNDListener
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    """

    # создание объекта задачи
    problem = Grishagin_mco(2, [2, 3])

    # Формируем параметры решателя
    params = SolverParameters(r=2, eps=0.01, iters_limit=1600, number_of_lambdas=1, start_lambdas=[[0, 1]], is_scaling=False)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Решение задачи
    sol = solver.solve()

    print("Grishagin_mco(2, [2, 3])")
    print("(r=2, eps=0.01, iters_limit=1600, number_of_lambdas=1, start_lambdas=[0, 1])")
    print("scaling false, new process")

    i=0
    x1 = [trial.point.float_variables[0] for trial in sol.best_trials]
    x2 = [trial.point.float_variables[1] for trial in sol.best_trials]
    print(x1)
    print(x2)
    #for k in range(problem.number_of_objectives):
    fv1 = [trial.function_values[0].value for trial in sol.best_trials]
    fv2 = [trial.function_values[1].value for trial in sol.best_trials]
    print(fv1)
    print(fv2)

    plt.plot(x1, x2, 'ro')
    plt.show()

    plt.plot(fv1, fv2, 'ro')
    plt.show()

