from Problems.mco_breast_cancer import mco_breast_cancer

from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    """
    Пробный пример многокритериальной оптимизации
    """

    # загрузка датасета
    X, y = load_breast_cancer(return_X_y=True)

    # разбиваем датасет на тренировочную и проверочную части
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # создание объекта задачи
    problem = mco_breast_cancer(X, y, X_train, y_train)
    # Формируем параметры решателя

    params = SolverParameters(r=3.0, eps=0.01, iters_limit=200, number_of_lambdas=50,
                              start_lambdas=[[0, 1]], is_scaling=False)

    # Создаем решатель
    solver = Solver(problem=problem, parameters=params)

    # Добавляем вывод резултатов в консоль
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    # Решение задачи
    sol = solver.solve()

    # вывод множества Парето (координаты - значения функций)
    var = [trial.point.float_variables for trial in sol.best_trials]
    val = [[-trial.function_values[i].value for i in range(2)] for trial in sol.best_trials]

    print("size pareto set: ", len(var))
    for fvar, fval in zip(var, val):
        print(fvar, fval)

    # Точки для постороения графика множества Парето y[0]-y[1]
    fv1 = [-trial.function_values[0].value for trial in sol.best_trials]
    fv2 = [-trial.function_values[1].value for trial in sol.best_trials]
    plt.plot(fv1, fv2, 'ro')
    plt.show()
