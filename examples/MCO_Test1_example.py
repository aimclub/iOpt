from problems.mco_test1 import mco_test1
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters

from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.output_system.listeners.static_painters import StaticPainterParetoListener
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Минимизация тестовой функции #1 многокритериальной оптимизации
    """

    # Создаем объект задачи
    problem = mco_test1()

    # Формируем параметры решателя
    params = SolverParameters(r=3.0, eps=0.01, iters_limit=16000, number_of_lambdas=50,
                              start_lambdas=[[0, 1]], is_scaling=False)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод результатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Добавляем построение графика множества Парето y[0]-y[1]
    sppl = StaticPainterParetoListener("mco_test1_pareto.png")
    solver.add_listener(sppl)

    # Решаем задачу
    sol = solver.solve()

    # Выводим множество Парето
    var = [trial.point.float_variables for trial in sol.best_trials]
    val = [[trial.function_values[i].value for i in range(2)] for trial in sol.best_trials]
    print("size pareto set: ", len(var))
    for fvar, fval in zip(var, val):
        print(fvar, fval)
